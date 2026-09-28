import json
from pathlib import Path

LLM_RESPONSES_ROOT = Path("app") / "services" / "llm_responses"


class LLMService:
    """
    We run the pipeline in advance to only use its responses.
    This ensures lightweight service that demonstrates how would the methods respond.
    """

    def __init__(self):

        with open(LLM_RESPONSES_ROOT / "llm-wo-cot-eval.json", "r") as f:
            self.responses_without_cot = json.load(f)["details"]

        with open(LLM_RESPONSES_ROOT / "llm-cot-eval.json", "r") as f:
            self.responses_with_cot = json.load(f)["details"]

        with open(LLM_RESPONSES_ROOT / "llm-cot-predictions.json", "r") as f:
            self.responses_with_cot_reasoning = json.load(f)

    def response_without_cot(self, item_id: str) -> dict:
        for response in self.responses_without_cot:
            if str(response["id"]) == item_id:
                return {"predicted_answer": response['prediction'], "acc": 1.0 if response['acc'] else 0.0,
                        "f1": response['f1']}
        return None

    def response_with_cot(self, item_id: str) -> dict:
        for response in self.responses_with_cot:
            if str(response["id"]) == item_id:
                # more controlled extraction is needed but our examples are hardcoded
                reasoning = json.loads(self.responses_with_cot_reasoning[item_id])["Reason"]
                return {"predicted_answer": response['prediction'], "acc": 1.0 if response['acc'] else 0.0,
                        "f1": response['f1'], "reasoning": reasoning}
        return None


def get_llm_service() -> LLMService:
    """FastAPI callable dependency"""
    return LLMService()
