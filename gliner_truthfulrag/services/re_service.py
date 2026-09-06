from abc import ABC, abstractmethod
from pathlib import Path
from openai import OpenAI

from gliner_truthfulrag.core.env_manager import settings

from concurrent.futures import ThreadPoolExecutor


class AbstractREService(ABC):
    @abstractmethod
    def predict_batch(self, nodes_per_chunk: dict) -> dict:
        ...


class REService(AbstractREService):
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _extract_chunk(self):
        pass

    def predict_batch(self, nodes_per_chunk: dict) -> dict:
        prompt_path = Path(settings.PROMPT_DIR) / "re_prompt.md"

        try:
            with open(prompt_path, "r") as f:
                PROMPT = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
