from abc import ABC, abstractmethod

from collections import defaultdict
import unicodedata


class AbstractDedupService(ABC):
    @abstractmethod
    def dedup_nodes(self, nodes: dict) -> dict:
        ...

    @abstractmethod
    def dedup_edges(self):
        ...


def _normalize_entity(name: str) -> str:
    return (
        unicodedata.normalize("NFKC", name)
        .replace("’", "'")
        .strip()
        .upper()
    )


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

    def dedup_edges(self):
        pass
