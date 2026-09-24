import argparse
import json
import asyncio
from time import perf_counter

from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pipeline-mode", choices=["kg", "direct-paths"], default="kg")
    parser.add_argument("--kg-backend", choices=["gliner", "llm"], default="llm")
    parser.add_argument(
        "--entropy-filter-method",
        choices=["legacy", "paper"],
        default="legacy",
    )
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument(
            "--generation-context",
            choices=["original", "none"],
            default="original",
    
        )
    parser.add_argument("--dataset-id", default="timeqa_2022_nota.json")
    parser.add_argument("--dataset-limit", type=int)
    return parser.parse_args()


async def run_experiment(args: argparse.Namespace):
    if args.dataset_limit is not None and args.dataset_limit < 1:
        raise ValueError("dataset-limit must be greater than zero")

    from truthfulrag.pipeline import TruthfulRAG

    dataset = load_dataset("json", data_files=f"datas/{args.dataset_id}")
    if len(dataset) < 1:
        raise ValueError(f"dataset {args.dataset_id} is empty")

    dataset = dataset["train"]
    if args.dataset_limit is not None:
        dataset = dataset.select(range(min(args.dataset_limit, len(dataset))))

    rag = TruthfulRAG(
        dataset=dataset,
        backend_type="hf",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        similarity_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=args.threshold,
        entropy_filter_method=args.entropy_filter_method,
        kg_backend=args.kg_backend if args.pipeline_mode == "kg" else "llm",    # if using direct path extraction, enforce llm backend
        gliner_kg_service_config={
            "gliner_model_id": "knowledgator/gliner-relex-large-v0.5",
            "llm_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "gliner_batch_size": 32,
            "llm_batch_size": 4
        },
        working_dir=f"./cache/{args.experiment_id}",
        output_dir=f"./outputs/{args.experiment_id}",
    )

    direct_path_service = None
    if args.pipeline_mode == "direct-paths":
        from gliner_truthfulrag.services.direct_path_service import QwenDirectPathService
        from gliner_truthfulrag.services.helpers import download_load_qwen

        # Load the direct path service
        model, tokenizer = download_load_qwen("Qwen/Qwen2.5-7B-Instruct")
        direct_path_service = QwenDirectPathService(model=model, tokenizer=tokenizer)

    # warmup-job
    ## initialize model setups for fair comparison
    ## if implementations are repeatedly re-loading weights after the warmup, this is included in the timings!!
    ## warmup is not measured
    if args.pipeline_mode == "kg":
        await rag.make_knowledge_graph(dataset[0])
        elements = await rag.knowledge_graph_retrieve(dataset[0])
    else:
        try:
            elements = direct_path_service.extract_direct_paths(dataset[0])
        except (ValueError, RuntimeError) as error:
            elements = []
            print(f"Direct-path warmup extraction failed: {error}")
    
    await rag.entropy_based_filter(
        sample=dataset[0],
        elements=elements[:30],
        threshold=args.threshold,
    )

    # eval on dataset
    with open(f"./outputs/{args.experiment_id}/stats.jsonl", "w") as f:
        filtered_list_full = []
        for idx, item in enumerate(tqdm(dataset)):
            single_dataset = dataset.select([idx])

            if args.pipeline_mode == "kg":
                start_kg_building = perf_counter()

                await rag.make_knowledge_graph(item)

                runtime_kg_building = perf_counter() - start_kg_building

                start_kg_retrieval = perf_counter()

                elements = await rag.knowledge_graph_retrieve(item)

                runtime_kg_retrieval = perf_counter() - start_kg_retrieval

                runtime_path_extraction = 0.0
                extraction_error = None

            # For direct extraction of paths kg building is skipped    
            else:
                runtime_kg_building = 0.0
                runtime_kg_retrieval = 0.0

                start = perf_counter()
                try:
                    elements = direct_path_service.extract_direct_paths(item)
                    extraction_error = None
                except (ValueError, RuntimeError) as error:
                    elements = []
                    extraction_error = str(error)
                runtime_path_extraction = perf_counter() - start

            start_entropy_filter = perf_counter()

            filtered_elements = await rag.entropy_based_filter(
                sample=item,
                elements=elements[:30],
                threshold=args.threshold,
            )

            filtered_list = [{"id": item['id'], "element": filtered_elements[:10]}]
            filtered_list_full.extend(filtered_list)

            runtime_entropy_filter = perf_counter() - start_entropy_filter

            start_pred_eval = perf_counter()

            predictions = await rag.get_predictions(single_dataset, filtered_list, generation_type="cot", generation_context=args.generation_context)

            results = rag.evaluate(single_dataset, predictions, cot_format=True, detailed_output=True)

            runtime_pred_eval = perf_counter() - start_pred_eval

            metrics = {
                "pipeline_mode": args.pipeline_mode,
                "runtime_entropy_filter": runtime_entropy_filter,
                "runtime_pred_eval": runtime_pred_eval,
                "elements_list": elements,
                "filtered_elements_list": filtered_elements,
            }
            # Make metrics dependent on the pipeline mode
            if args.pipeline_mode == "kg":
                metrics.update({
                    "runtime_kg_building": runtime_kg_building,
                    "runtime_kg_retrieval": runtime_kg_retrieval,
                })
            else:
                metrics.update({
                    "runtime_path_extraction": runtime_path_extraction,
                    "extraction_error": extraction_error,
                })

            results.update(metrics)

            f.write(json.dumps(results) + "\n")

    with open(f"./outputs/{args.experiment_id}/res.json", "w") as f:
        predictions_full = await rag.get_predictions(dataset, filtered_list_full, generation_type="cot", generation_context=args.generation_context)
        results = rag.evaluate(dataset, predictions_full, cot_format=True, detailed_output=True)

        json.dump(results, f, indent=2)

    # clean-up (GLiNER)
    if rag._gliner_kg_service is not None:
        from gliner_truthfulrag.services.helpers import unload_model

        unload_model(rag._gliner_kg_service.qwen_service.model)
        unload_model(rag._gliner_kg_service.gliner_service.model)

    # clean-up (Direct Path Service)
    if direct_path_service is not None:
        from gliner_truthfulrag.services.helpers import unload_model

        unload_model(direct_path_service.model)


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(run_experiment(cli_args))
