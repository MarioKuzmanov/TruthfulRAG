from abc import ABC, abstractmethod

from gliner_truthfulrag.services.chunk_service import ChunkerService
from gliner_truthfulrag.services.llm_service import QwenLLMService
from gliner_truthfulrag.services.ner_re_service import GLiNERService
from gliner_truthfulrag.services.helpers import *

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)


class AbstractKGService(ABC):
    @abstractmethod
    def chunk_step(self, item: dict):
        ...

    @abstractmethod
    def llm_step(self, chunks: list[dict]):
        ...

    @abstractmethod
    def ner_re_step(self, chunks: list[dict]):
        ...

    @abstractmethod
    def build_kg_step(self, item: dict) -> tuple[list[dict], list[dict]]:
        ...

class KGService(AbstractKGService):

    def __init__(self, gliner_model_id: str, llm_model_id: str, gliner_batch_size: int = 32, llm_batch_size: int = 4):
        # download and load models
        gliner_model, gliner_tokenizer, llm_model, llm_tokenizer = init_pipeline(gliner_model_id, llm_model_id)

        # load services
        self.chunker = ChunkerService(tokenizer=gliner_tokenizer, max_tokens=512, offset=16, overlap_tokens=32)
        self.qwen_service = QwenLLMService(model=llm_model, tokenizer=llm_tokenizer, batch_size=llm_batch_size)

        # default entity labels
        entity_labels = ["organization", "person", "location", "event"]
        self.gliner_service = GLiNERService(model=gliner_model, entity_labels=entity_labels, relation_labels=[],
                                            batch_size=gliner_batch_size)

    def chunk_step(self, item: dict) -> list[dict]:
        chunks = self.chunker.chunk_by_gliner_token_size(item)
        return chunks

    def llm_step(self, chunks: list[dict]) -> list[str]:
        relation_labels = self.qwen_service.extract_raw_predicates(chunks)
        return relation_labels

    def ner_re_step(self, chunks: list[dict]) -> tuple[dict, list[dict]]:
        nodes, relations = self.gliner_service.predict_batch(chunks)
        return nodes, relations

    # best-effort conformance to TruthfulRAG entities and relations output
    def build_kg_step(self, item: dict) -> tuple[list[dict], list[dict]]:
        # STEP 1: Chunking
        chunks = self.chunk_step(item)

        # STEP 2: LLM raw predicate extraction
        relation_labels = self.llm_step(chunks)

        self.gliner_service.relation_labels = relation_labels

        # STEP 3: GLiNER NER and RE
        nodes, relations = self.ner_re_step(chunks)

        logger.info(f"KG Building step completed")

        return nodes, relations
