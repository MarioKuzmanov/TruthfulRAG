from abc import ABC, abstractmethod
from pathlib import Path
from openai import OpenAI
import os

from gliner_truthfulrag.core.env_manager import settings
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)

from concurrent.futures import ThreadPoolExecutor, as_completed


class AbstractREService(ABC):
    @abstractmethod
    def predict_batch(self, chunks: list[dict], nodes_per_chunk: dict) -> dict:
        ...


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class REOutput(BaseModel):
    triples: list[Triple]


# being careful with API rate limits
NUM_PARALLEL_WORKERS = min(8, os.cpu_count() or 1)


class REService(AbstractREService):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt_path = Path(settings.PROMPT_DIR) / "re_prompt.md"

        try:
            with open(prompt_path, "r") as f:
                self.prompt = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    def _extract_chunk(self, chunk: dict, nodes_in_chunk: list):
        context, chunk_id = chunk["context"], chunk["chunk_id"]
        entities, allowed_entities = [], set()
        for node in nodes_in_chunk:
            entities.append({
                "name": node["node"],
                "type": node["node_type"],
            })
            allowed_entities.add(node["node"])

        logger.info(f"Chunk_id={chunk_id}, entities={entities}")

        if len(entities) < 2:
            return []

        prompt_input = f"""SOURCE TEXT: {context}\nENTITIES: {entities}"""

        response = self.client.responses.parse(
            model=self.model_id,
            input=[
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": prompt_input,
                },
            ],
            text_format=REOutput,
        )

        parsed = response.output_parsed

        if parsed is None:
            logger.warning(
                "No parsed RE output for chunk_id=%s",
                chunk_id,
            )
            return []

        # we can be more harsh if we define domain and range
        triples = [
            {
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "chunk_id": chunk_id,
            }
            for triple in parsed.triples if triple.subject in allowed_entities and triple.object in allowed_entities
        ]

        logger.info(
            "RE chunk_id=%s triples=%s",
            chunk_id,
            triples,
        )

        return triples

    def predict_batch(self, chunks: list[dict], nodes_per_chunk: dict) -> dict:
        relations_per_chunk = {}

        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            future_to_chunk_id = {
                executor.submit(
                    self._extract_chunk,
                    chunk,
                    nodes_per_chunk.get(chunk["chunk_id"], []),
                ): chunk["chunk_id"]
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk_id):
                chunk_id = future_to_chunk_id[future]

                try:
                    relations_per_chunk[chunk_id] = future.result()

                except Exception:
                    logger.exception(
                        "RE failed for chunk_id=%s",
                        chunk_id,
                    )
                    relations_per_chunk[chunk_id] = []

        return relations_per_chunk
