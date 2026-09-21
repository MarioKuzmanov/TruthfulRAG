from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)


class AbstractNerREService(ABC):
    @abstractmethod
    def predict_batch(self, chunks: list) -> dict:
        ...


class GLiNERService(AbstractNerREService):
    def __init__(self, model: Any, entity_labels: list[str], relation_labels: list[str], batch_size: int) -> None:
        self.model = model
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.batch_size = batch_size

    def predict_batch(self, chunks: list) -> dict:
        # batch-inference on GPU
        # chunk = {item_id: str, tokens: int, chunk_id: str, context: str}

        texts = [chunk["context"] for chunk in chunks]

        # order is same as in chunks
        pred_entities, pred_relations = self.model.inference(texts, labels=self.entity_labels + ["other"],
                                                             relations=self.relation_labels, threshold=0.4,
                                                             adjacency_threshold=0.4, relation_threshold=0.8,
                                                             batch_size=self.batch_size, multi_label=False,
                                                             return_relations=True, flat_ner=True)

        if len(pred_entities) != len(pred_relations) or len(pred_entities) != len(chunks):
            raise AssertionError("chunks and preds do not match")

        # map to KG nodes
        nodes, relations = defaultdict(list), []
        for i, (text_entities, text_relations) in enumerate(zip(pred_entities, pred_relations)):
            chunk = chunks[i]

            for e in text_entities:
                node = e["text"].upper()

                node_meta = {"entity_name": e["text"], "entity_type": e["label"].upper(), "source_id": chunk["item_id"],
                             "chunk_id": chunk["chunk_id"],
                             "confidence": e["score"]}

                nodes[node].append(node_meta)

            for rel in text_relations:
                # head
                head_node = rel["head"]
                head_node_meta = {"entity_name": head_node["text"], "entity_type": head_node["type"].upper(),
                                  "source_id": chunk["item_id"],
                                  "chunk_id": chunk["chunk_id"]}

                # tail
                tail_node = rel["tail"]
                tail_node_meta = {"entity_name": tail_node["text"], "entity_type": tail_node["type"].upper(),
                                  "source_id": chunk["item_id"],
                                  "chunk_id": chunk["chunk_id"]}

                relation, score = rel["relation"], rel["score"]

                relations.append({"head": head_node_meta, "tail": tail_node_meta, "relation": relation, "score": score})

        logger.info(f"extracted KG:\nnodes={nodes}\nedges={relations}\n")

        return nodes, relations
