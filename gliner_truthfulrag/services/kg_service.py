from abc import ABC, abstractmethod
import os

from gliner_truthfulrag.services.chunk_service import ChunkerService
from gliner_truthfulrag.services.ner_service import GLiNERService
from gliner_truthfulrag.services.dedup_service import DedupService
from gliner_truthfulrag.services.re_service import REService

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)

# GLiNER labels and threshold (balance precision and recall)
LABELS = ["organization", "person", "location", "event"]
THRESHOLD = 0.5


class AbstractKGService(ABC):
    @abstractmethod
    def chunk_step(self, item: dict):
        ...

    @abstractmethod
    def ner_step(self, item: dict):
        ...

    @abstractmethod
    def re_step(self, nodes_per_chunk: dict):
        ...

    @abstractmethod
    def dedup_step(self, nodes: dict, to_dedup_nodes: bool):
        ...

    @abstractmethod
    def build_kg(self, item: dict):
        ...


NUM_PARALLEL_WORKERS = os.cpu_count() or 1


class KGService(AbstractKGService):

    def __init__(self, ner_model_id: str, re_model_id: str):
        self.re_model_id = re_model_id

        # load services on startup
        self.ner_service = GLiNERService(model_id=ner_model_id, labels=LABELS, threshold=THRESHOLD)

        tokenizer = self.ner_service.model.data_processor.transformer_tokenizer
        self.chunker_service = ChunkerService(tokenizer=tokenizer)

        self.dedup_service = DedupService()

        self.re_service = REService()

    def chunk_step(self, item: dict) -> list[dict]:

        chunks = self.chunker_service.chunk_by_token_size(item=item)

        logger.info(f"chunker response: {chunks}\n\n")

        return chunks

    def ner_step(self, item: dict) -> tuple[list, list]:

        # STEP 1: Chunk
        chunks = self.chunk_step(item=item)

        # STEP 2: NER
        nodes = self.ner_service.predict_batch(chunks)

        return chunks, nodes

    def re_step(self, nodes_per_chunk: dict):
        self.re_service.predict_batch(nodes_per_chunk)

    def dedup_step(self, nodes: dict, to_dedup_nodes: bool):

        if to_dedup_nodes:
            nodes_per_chunk = self.dedup_service.dedup_nodes(nodes)
            return nodes_per_chunk
        else:
            self.dedup_service.dedup_edges()

        return {}

    def build_kg(self, item: dict):

        # STEP 1: Chunking and NER
        chunks, nodes = self.ner_step(item)

        # STEP 2: Per Chunk Nodes Dedup
        nodes_per_chunk = self.dedup_step(nodes=nodes, to_dedup_nodes=True)

        # STEP 3: RE Per Chunk
        self.re_step(nodes_per_chunk=nodes_per_chunk)

        # STEP 4: Per Chunk RE Dedup

        # STEP 5: KG ensemble
