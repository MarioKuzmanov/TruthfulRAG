from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
import os

from gliner import GLiNER
from gliner.model import UniEncoderSpanGLiNER

from gliner_truthfulrag.core.env_manager import settings
import torch
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)

BATCH_SIZE = 16


class AbstractNerService(ABC):
    @abstractmethod
    def load(self, model_id: str) -> Any:
        ...


class GLiNERService(AbstractNerService):
    def __init__(self, model_id: str, labels: list[str], threshold: float) -> None:
        self.model = self.load(model_id)
        self.labels = labels
        self.threshold = threshold

    def load(self, model_id: str) -> UniEncoderSpanGLiNER:

        # load optimized GLiNER
        os.environ["USE_FLASHDEBERTA"] = settings.USE_FLASHDEBERTA

        if not torch.cuda.is_available():
            raise AttributeError(f"GPU: {settings.MAP_LOCATION} not available. Loading on CPU instead...")

        if "gliner" not in model_id:
            raise NotImplementedError(f"Model {model_id} not implemented.")

        m = GLiNER.from_pretrained(model_id, map_location=settings.MAP_LOCATION)
        encoder = m.model.token_rep_layer.bert_layer.model

        logger.info(f"type GLiNER: {type(m)}, type Encoder: {type(encoder)}")

        m.compile()

        m.half().eval()

        return m

    def unload(self):
        del self.model

        torch.cuda.empty_cache()

    def predict_batch(self, chunks: list) -> dict:
        # batch-inference on GPU
        # chunk = {item_id: str, tokens: int, chunk_id: str, context: str}

        texts = [chunk["context"] for chunk in chunks]

        # order is same as in chunks
        preds = self.model.inference(texts, labels=self.labels, threshold=self.threshold, batch_size=BATCH_SIZE)

        if len(preds) != len(chunks):
            raise AssertionError("chunks and preds do not match")

        # map to KG nodes
        nodes = defaultdict(list)
        for chunk, entities in zip(chunks, preds):
            for e in entities:
                node = e['text'].upper()
                node_type = e['label'].upper()
                node_meta = {"entity_name": e["text"], "entity_type": node_type, "source_id": chunk["item_id"],
                             "chunk_id": chunk["chunk_id"],
                             "confidence": e["score"]}
                nodes[node].append(node_meta)

        logger.info(f"extracted nodes=%s", nodes)

        return nodes
