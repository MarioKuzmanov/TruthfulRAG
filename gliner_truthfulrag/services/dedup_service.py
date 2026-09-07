from abc import ABC, abstractmethod

from collections import defaultdict
import unicodedata

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)


class AbstractDedupService(ABC):
    @abstractmethod
    def dedup_nodes(self, nodes: dict) -> dict:
        ...

    @abstractmethod
    def dedup_edges(self, edges: dict) -> list[dict]:
        ...


def _normalize_entity(name: str) -> str:
    return (
        unicodedata.normalize("NFKC", name)
        .replace("’", "'")
        .strip()
        .upper()
    )


def _normalize_predicate(predicate: str) -> str:
    return (
        predicate
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _dedup_edges_logic(chunk_id: str, dedup_edges: dict, triples: list) -> None:
    for triple in triples:

        subject = _normalize_entity(triple["subject"])
        predicate = _normalize_predicate(triple["predicate"])
        object_ = _normalize_entity(triple["object"])

        key = (
            subject,
            predicate,
            object_,
        )

        if key not in dedup_edges:
            dedup_edges[key] = {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "chunk_ids": set(),
            }

        dedup_edges[key]["chunk_ids"].add(chunk_id)


class DedupService(AbstractDedupService):
    def dedup_nodes(self, nodes: dict) -> dict:

        nodes_dedup = defaultdict(lambda: defaultdict(list))

        for entity, mentions in nodes.items():
            normalized_entity = _normalize_entity(entity)

            for mention in mentions:
                nodes_dedup[mention["chunk_id"]][normalized_entity].append(
                    (
                        mention["confidence"],
                        mention["entity_type"],
                    )
                )

        nodes_per_chunk = defaultdict(list)

        for chunk_id, entities in nodes_dedup.items():
            for entity, scores_types in entities.items():

                type_scores = defaultdict(float)

                for score, entity_type in scores_types:
                    type_scores[entity_type] += score

                best_type = max(
                    type_scores,
                    key=type_scores.get,
                )

                selected_scores = [
                    score
                    for score, entity_type in scores_types
                    if entity_type == best_type
                ]

                nodes_per_chunk[chunk_id].append({
                    "node": entity,
                    "node_type": best_type,
                    "confidence": sum(selected_scores) / len(selected_scores),
                    "max_confidence": max(selected_scores),
                    "mentions": len(selected_scores),
                })

        return nodes_per_chunk

    def dedup_edges(self, edges: dict) -> list[dict]:
        dedup_edges = {}

        for chunk_id, triples in edges.items():
            _dedup_edges_logic(chunk_id, dedup_edges, triples)

        edges_clean = []

        for edge in dedup_edges.values():
            edge["chunk_ids"] = sorted(edge["chunk_ids"])
            edges_clean.append(edge)

        logger.info(f"Dedup Edges={edges_clean}")

        return edges_clean

    def dedup_edges_final(self, edges: list[dict]):
        dedup_edges = {}
        for e in edges:
            subject = _normalize_entity(e["subject"])
            predicate = _normalize_predicate(e["predicate"])
            object_ = _normalize_entity(e["object"])
            chunk_ids = e.get("chunk_ids", [])

            key = (
                subject,
                predicate,
                object_,
            )

            if key not in dedup_edges:
                dedup_edges[key] = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "chunk_ids": set(),
                }

            dedup_edges[key]["chunk_ids"].update(chunk_ids)

        kg = [{**edge, "chunk_ids": sorted(edge["chunk_ids"])} for edge in dedup_edges.values()]

        logger.info(f"KG={kg}")

        return kg
