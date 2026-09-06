from tqdm import tqdm
import json

from gliner_truthfulrag.services.kg_service import KGService, AbstractKGService


def run_kg(item: dict, service: AbstractKGService) -> None:
    service.build_kg(item=item)


if __name__ == "__main__":

    DATASET_ID = "timeqa_2022_nota.json"

    try:
        with open(f"datas/{DATASET_ID}", "r") as f:
            dataset = json.load(f)
            dataset = [dataset[2]]

    except Exception:
        raise f"{DATASET_ID} not found"

    kg_builder = KGService(ner_model_id="gliner-community/gliner_small-v2.5", re_model_id="gpt-4o-mini")

    for item in tqdm(dataset):
        run_kg(item=item, service=kg_builder)
