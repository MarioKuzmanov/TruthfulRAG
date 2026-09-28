import json
from pathlib import Path
import os
from collections import defaultdict

RAG_TRIPLES_RESPONSES_ROOT = Path("app") / "services" / "rag_triples_responses"


class RAGTriplesService:
    def __init__(self):
        self.extracted_paths = defaultdict(dict)
        per_item = os.listdir(RAG_TRIPLES_RESPONSES_ROOT / "per_item")

        for item_file in per_item:
            item_path = str(RAG_TRIPLES_RESPONSES_ROOT / "per_item" / item_file)

            item_id = str(item_file[: item_file.find("-")].strip())

            with open(item_path, "r") as f:
                item = json.load(f)
                if "-filtered-elements" in item_path:
                    self.extracted_paths[item_id]["filtered_paths"] = item
                elif "-elements" in item_path:
                    self.extracted_paths[item_id]["paths"] = item
                else:
                    continue

        ## wo context
        with open(RAG_TRIPLES_RESPONSES_ROOT / "rag-triples-wo-context-eval.json", "r") as f:
            self.response_wo_context = json.load(f)["details"]

        with open(RAG_TRIPLES_RESPONSES_ROOT / "rag-triples-wo-context-predictions.json", "r") as f:
            self.reasoning_wo_context = json.load(f)

        ## with context
        with open(RAG_TRIPLES_RESPONSES_ROOT / "rag-triples-context-eval.json", "r") as f:
            self.response_with_context = json.load(f)["details"]

        with open(RAG_TRIPLES_RESPONSES_ROOT / "rag-triples-context-predictions.json", "r") as f:
            self.reasoning_with_context = json.load(f)

    def get_paths(self, item_id: str):
        for curr_item_id in self.extracted_paths:
            if curr_item_id == item_id:
                return {"extracted_paths": self.extracted_paths[item_id]["paths"],
                        "filtered_paths": self.extracted_paths[item_id]["filtered_paths"]}
        return None

    def get_response_wo_context(self, item_id: str):
        for item in self.response_wo_context:
            if str(item["id"]) == item_id:
                reasoning = json.loads(self.reasoning_wo_context[item_id])["Reason"]
                return {"predicted_answer": item['prediction'], "acc": 1.0 if item['acc'] else 0.0,
                        "f1": item['f1'], "reasoning": reasoning}
        return None

    def get_response_with_context(self, item_id: str):
        for item in self.response_with_context:
            if str(item["id"]) == item_id:
                reasoning = json.loads(self.reasoning_with_context[item_id])["Reason"]
                return {"predicted_answer": item['prediction'], "acc": 1.0 if item['acc'] else 0.0,
                        "f1": item['f1'], "reasoning": reasoning}
        return None


def get_rag_triples_service() -> RAGTriplesService:
    """FastAPI callable dependency"""
    return RAGTriplesService()
