import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import asyncio

import json
import argparse
from time import perf_counter

from truthfulrag.pipeline import TruthfulRAG
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset-id", default="timeqa_2022_nota.json")
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--use-rag", action="store_true", help="--use-rag means `RAG`, without use-rag means `w/o RAG`")
    parser.add_argument("--use-cot", action="store_true", help="--use-cot means `cot`, without use-cot means `wo_cot`")
    return parser.parse_args()


async def run_experiment(args: argparse.Namespace):
    if args.dataset_limit is not None and args.dataset_limit < 1:
        raise ValueError("dataset-limit must be greater than zero")

    try:
        dataset = load_dataset("json", data_files=f"datas/{args.dataset_id}")
        if len(dataset) < 1:
            raise ValueError(f"dataset {args.dataset_id} is empty")

        dataset = dataset['train']
        if args.dataset_limit is not None:
            dataset = dataset.select(range(min(args.dataset_limit, len(dataset))))

    except Exception:
        raise f"{args.dataset_id} not found"

    ## the backends for baseline experiments are not configurable
    rag = TruthfulRAG(
        dataset=dataset,
        backend_type="hf",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        similarity_model="sentence-transformers/all-MiniLM-L6-v2",
        entropy_filter_method="legacy",
        threshold=3,
        kg_backend="llm",
        working_dir=f"./cache/{args.experiment_id}",
        output_dir=f"./outputs/{args.experiment_id}",
    )

    with open(f"./outputs/{args.experiment_id}/res.json", "w") as f:
        start = perf_counter()

        predictions = await rag.get_predictions_wo_elements(dataset, withrag=args.use_rag,
                                                            generation_type="cot" if args.use_cot else "wo_cot")
        results = rag.evaluate(dataset, predictions, cot_format=args.use_cot, detailed_output=True)

        duration = perf_counter() - start

        metrics = {"runtime_pred_eval": duration}
        results.update(metrics)

        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_experiment(parse_args()))
