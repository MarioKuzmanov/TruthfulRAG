from abc import ABC, abstractmethod
import os

from gliner_truthfulrag.services.chunk_service import ChunkerService
from gliner_truthfulrag.services.ner_service import GLiNERService
from gliner_truthfulrag.services.dedup_service import DedupService
from gliner_truthfulrag.services.re_service import REService
from gliner_truthfulrag.services.canonicalize_service import CanonicalizeService

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
GLINER_THRESHOLD = 0.7


class AbstractKGService(ABC):
    @abstractmethod
    def chunk_step(self, item: dict):
        ...

    @abstractmethod
    def ner_step(self, item: dict):
        ...

    @abstractmethod
    def re_step(self, chunks: list[dict], nodes_per_chunk: dict):
        ...

    @abstractmethod
    def dedup_nodes_step(self, nodes: dict):
        ...

    @abstractmethod
    def dedup_edges_step(self, edges: dict):
        ...

    @abstractmethod
    def canonicalize_step(self, nodes: dict, dedup_edges: list[dict]):
        ...

    @abstractmethod
    def build_kg(self, item: dict):
        ...


NUM_PARALLEL_WORKERS = os.cpu_count() or 1


class KGService(AbstractKGService):

    def __init__(self, ner_model_id: str, re_model_id: str, embedding_model_id: str, threshold: int | float):
        self.re_model_id = re_model_id

        # load services on startup
        self.ner_service = GLiNERService(model_id=ner_model_id, labels=LABELS, threshold=GLINER_THRESHOLD)

        tokenizer = self.ner_service.model.data_processor.transformer_tokenizer
        self.chunker_service = ChunkerService(tokenizer=tokenizer)

        self.dedup_service = DedupService()

        self.re_service = REService(model_id=re_model_id)

        self.canonicalize_service = CanonicalizeService(embedding_model_id=embedding_model_id, threshold=threshold)

    def chunk_step(self, item: dict) -> list[dict]:
        chunks = self.chunker_service.chunk_by_token_size(item=item)

        logger.info(f"chunker response: {chunks}\n\n")

        return chunks

    def ner_step(self, item: dict) -> tuple[list, list]:
        # STEP: Chunk
        chunks = self.chunk_step(item=item)

        # STEP: NER
        nodes = self.ner_service.predict_batch(chunks)

        return chunks, nodes

    def re_step(self, chunks: list[dict], nodes_per_chunk: dict) -> dict:
        # STEP: RE
        triples_per_chunk = self.re_service.predict_batch(chunks, nodes_per_chunk)
        return triples_per_chunk

    def dedup_nodes_step(self, nodes: dict) -> dict:
        # STEP: Dedup Nodes
        nodes_per_chunk = self.dedup_service.dedup_nodes(nodes)
        return nodes_per_chunk

    def dedup_edges_step(self, edges: dict) -> list[dict]:
        # STEP: Dedup Edges
        dedup_all_edges = self.dedup_service.dedup_edges(edges)
        return dedup_all_edges

    def dedup_edges_final(self, edges: list[dict]):
        # STEP: Dedup canonical edges
        dedup_canonical_edges = self.dedup_service.dedup_edges_final(edges)
        return dedup_canonical_edges

    def canonicalize_step(self, nodes: dict, dedup_edges: list[dict]) -> list[dict]:
        # STEP: KG Canonicalization
        canonicalized_edges = self.canonicalize_service.canonicalize_edges(nodes, dedup_edges)
        return canonicalized_edges

    def build_kg(self, item: dict):
        # STEP 1: Chunking and NER
        chunks, nodes = self.ner_step(item)

        # STEP 2: Per Chunk Nodes Dedup
        nodes_per_chunk = self.dedup_nodes_step(nodes=nodes)

        # STEP 3: RE Per Chunk
        edges_per_chunk = self.re_step(chunks=chunks, nodes_per_chunk=nodes_per_chunk)

        # STEP 4: All Edges Dedup
        edges_dedup = self.dedup_edges_step(edges=edges_per_chunk)

        # STEP 5: Canonicalize KG
        edges_canonicalize = self.canonicalize_step(nodes=nodes_per_chunk, dedup_edges=edges_dedup)

        # STEP 6: Form Clean KG
        kg = self.dedup_edges_final(edges=edges_canonicalize)

        return kg
