import os
import json
from tqdm import tqdm
from gliner_truthfulrag.core.env_manager import settings

os.environ["USE_FLASHDEBERTA"] = settings.USE_FLASHDEBERTA

import asyncio

from gliner_truthfulrag.services.kg_service import AbstractKGService
from truthfulrag.pipeline import TruthfulRAG
from time import perf_counter

from datasets import load_dataset


def run_kg(item: dict, service: AbstractKGService) -> tuple[dict, list[dict]]:
    return service.build_kg_step(item=item)


async def run_experiment():
    DATASET_ID = "timeqa_2022_nota.json"
    EXPERIMENT_ID = "gliner"

    try:
        dataset = load_dataset("json", data_files=f"datas/{DATASET_ID}")
        dataset = dataset['train']
        dataset = dataset.select(range(0, 5, 1))

    except Exception:
        raise f"{DATASET_ID} not found"

    rag = TruthfulRAG(
        dataset=dataset,
        backend_type="hf",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        similarity_model="sentence-transformers/all-MiniLM-L6-v2",
        threshold=3,
        kg_backend="gliner",
        gliner_kg_service_config={
            "gliner_model_id": "knowledgator/gliner-relex-large-v0.5",
            "llm_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "gliner_batch_size": 32,
            "llm_batch_size": 4
        },
        working_dir=f"./cache/{EXPERIMENT_ID}",
        output_dir=f"./outputs/{EXPERIMENT_ID}",
    )

    # eval on dataset
    with open(f"./outputs/{EXPERIMENT_ID}/stats.jsonl", "w") as f:
        filtered_list_full = []
        for idx, item in enumerate(tqdm(dataset)):
            single_dataset = dataset.select([idx])
            start_kg_building = perf_counter()

            await rag.make_knowledge_graph(item)

            runtime_kg_building = perf_counter() - start_kg_building

            start_kg_retrieval = perf_counter()

            elements = await rag.knowledge_graph_retrieve(item)

            runtime_kg_retrieval = perf_counter() - start_kg_retrieval

            start_entropy_filter = perf_counter()

            filtered_elements = await rag.entropy_based_filter(sample=item, elements=elements[:30], threshold=3)

            filtered_list = [{"id": item['id'], "element": filtered_elements[:10]}]
            filtered_list_full.extend(filtered_list)

            runtime_entropy_filter = perf_counter() - start_entropy_filter

            start_pred_eval = perf_counter()

            predictions = await rag.get_predictions(single_dataset, filtered_list, generation_type="cot")

            results = rag.evaluate(single_dataset, predictions, cot_format=True, detailed_output=True)

            runtime_pred_eval = perf_counter() - start_pred_eval

            metrics = {"runtime_kg_building": runtime_kg_building, "runtime_kg_retrieval": runtime_kg_retrieval,
                       "runtime_entropy_filter": runtime_entropy_filter, "runtime_pred_eval": runtime_pred_eval,
                       "elements_list": elements, "filtered_elements_list": filtered_elements}

            results.update(metrics)

            f.write(json.dumps(results) + "\n")

    with open(f"./outputs/{EXPERIMENT_ID}/res.json", "w") as f:
        predictions_full = await rag.get_predictions(dataset, filtered_list_full, generation_type="cot")
        results = rag.evaluate(dataset, predictions_full, cot_format=True, detailed_output=True)

        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_experiment())
