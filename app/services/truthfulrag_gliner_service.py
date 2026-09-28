import json
from pathlib import Path
from app.core.registry import REGISTERED_ITEMS_IDS

TRUTHFULRAG_GLINER_RESPONSES_ROOT = Path("app") / "services" / "truthfulrag_gliner_responses"


class TruthfulRAGGlinerService:
    def __init__(self):
        self.filtered_chunks = []
        self.extracted_entities = []
        self.extracted_relations = []
        self.paths = []

        for item_id in REGISTERED_ITEMS_IDS:
            item_id = str(item_id)

            chunks_path = TRUTHFULRAG_GLINER_RESPONSES_ROOT / "per_item" / f"{item_id}-truthfulrag-gliner-filtered-chunks.json"
            entities_path = TRUTHFULRAG_GLINER_RESPONSES_ROOT / "per_item" / f"{item_id}-truthfulrag-gliner-entities.json"
            relations_path = TRUTHFULRAG_GLINER_RESPONSES_ROOT / "per_item" / f"{item_id}-truthfulrag-gliner-relations.json"
            elements_path = TRUTHFULRAG_GLINER_RESPONSES_ROOT / "per_item" / f"{item_id}-truthfulrag-gliner-retrieved-query.json"
            filtered_elements_path = TRUTHFULRAG_GLINER_RESPONSES_ROOT / "per_item" / f"{item_id}-truthfulrag-gliner-filtered-elements.json"

            with open(chunks_path, "r") as f:
                item_chunks = json.load(f)
                self.filtered_chunks.append({"id": item_id, "chunks": item_chunks})

            with open(entities_path, "r") as f:
                item_entities = json.load(f)
                self.extracted_entities.append({"id": item_id, "entities": item_entities})

            with open(relations_path, "r") as f:
                item_relations = json.load(f)
                self.extracted_relations.append({"id": item_id, "relations": item_relations})

            with open(elements_path, "r") as f:
                elements = json.load(f)

            with open(filtered_elements_path, "r") as f:
                filtered_elements = json.load(f)

            self.paths.append({"id": item_id, "elements": elements, "filtered_elements": filtered_elements})

        ## without context
        with open(TRUTHFULRAG_GLINER_RESPONSES_ROOT / "truthfulrag-gliner-wo-context-eval.json", "r") as f:
            self.responses_wo_context = json.load(f)["details"]

        with open(TRUTHFULRAG_GLINER_RESPONSES_ROOT / "truthfulrag-gliner-wo-context-predictions.json", "r") as f:
            self.reasoning_wo_context = json.load(f)

        ## with context
        with open(TRUTHFULRAG_GLINER_RESPONSES_ROOT / "truthfulrag-gliner-context-eval.json", "r") as f:
            self.responses_with_context = json.load(f)["details"]

        with open(TRUTHFULRAG_GLINER_RESPONSES_ROOT / "truthfulrag-gliner-context-predictions.json", "r") as f:
            self.reasoning_with_context = json.load(f)

    def get_filtered_chunks(self, item_id: str):
        for chunk_dict in self.filtered_chunks:
            if item_id == chunk_dict["id"]:
                chunks = chunk_dict["chunks"]
                return chunks
        return None

    def get_extracted_entities(self, item_id: str):
        for entities_dict in self.extracted_entities:
            if item_id == entities_dict["id"]:
                entities = entities_dict["entities"]
                #  { ENTITY_NAME: {...} }
                return entities
        return None

    def get_extracted_relations(self, item_id: str):
        for relations_dict in self.extracted_relations:
            if item_id == relations_dict["id"]:
                relations = relations_dict["relations"]
                return relations
        return None

    def get_paths(self, item_id: str):
        for item_paths in self.paths:
            if item_id == item_paths["id"]:
                return item_paths
        return None

    def get_response_without_context(self, item_id: str):
        for item in self.responses_wo_context:
            if str(item["id"]) == item_id:
                reasoning = json.loads(self.reasoning_wo_context[item_id])["Reason"]
                return {"predicted_answer": item['prediction'], "acc": 1.0 if item['acc'] else 0.0,
                        "f1": item['f1'], "reasoning": reasoning}
        return None

    def get_response_with_context(self, item_id: str):
        for item in self.responses_with_context:
            if str(item["id"]) == item_id:
                reasoning = json.loads(self.reasoning_with_context[item_id])["Reason"]
                return {"predicted_answer": item['prediction'], "acc": 1.0 if item['acc'] else 0.0,
                        "f1": item['f1'], "reasoning": reasoning}
        return None


def get_truthfulrag_gliner_service() -> TruthfulRAGGlinerService:
    """FastAPI callable dependency"""
    return TruthfulRAGGlinerService()
