from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


class AbstractCanonicalizeService(ABC):
    @abstractmethod
    def canonicalize_edges(self, nodes: dict, edges: list[dict]) -> list[dict]:
        ...


def find(parents, p):
    while parents[p] != p:
        parents[p] = parents[parents[p]]
        p = parents[p]
    return p


def union(parents, predicate_freqs, p1, p2) -> None:
    r1 = find(parents, p1)
    r2 = find(parents, p2)

    if r1 != r2:
        if predicate_freqs[r1] >= predicate_freqs[r2]:
            parents[r2] = r1
        else:
            parents[r1] = r2


class CanonicalizeService(AbstractCanonicalizeService):

    def __init__(self, embedding_model_id: str, threshold: int | float) -> None:
        self.embedder = SentenceTransformer(embedding_model_id)
        self.threshold = threshold

    def canonicalize_edges(self, nodes: dict, edges: list[dict]) -> list[dict]:

        # entity -> possible types
        entity_types = defaultdict(set)

        for chunk_nodes in nodes.values():
            for node in chunk_nodes:
                entity_types[node["node"]].add(node["node_type"])

        # predicate -> observed domain/range types
        predicates_domain_range = defaultdict(
            lambda: {
                "domain": set(),
                "range": set(),
            }
        )

        predicate_freqs = defaultdict(int)

        for edge in edges:
            predicate = edge["predicate"]

            subject_types = entity_types.get(edge["subject"], set())
            object_types = entity_types.get(edge["object"], set())

            predicates_domain_range[predicate]["domain"].update(subject_types)

            predicates_domain_range[predicate]["range"].update(object_types)

            predicate_freqs[predicate] += len(edge.get("chunk_ids", [])) or 1

        predicates = list(predicates_domain_range.keys())

        reprs = []

        for predicate in predicates:
            signature = predicates_domain_range[predicate]

            subject_type_str = "/".join(sorted(signature["domain"]))

            object_type_str = "/".join(sorted(signature["range"]))

            representation = (
                f"{subject_type_str} "
                f"{predicate.replace('_', ' ')} "
                f"{object_type_str}"
            )

            reprs.append(representation)

        emb = self.embedder.encode(
            reprs,
            normalize_embeddings=True,
        )

        mat = emb @ emb.T

        canonical_map = {
            predicate: predicate
            for predicate in predicates
        }

        for i in range(len(predicates)):
            for j in range(i + 1, len(predicates)):

                p1 = predicates[i]
                p2 = predicates[j]

                sig1 = predicates_domain_range[p1]
                sig2 = predicates_domain_range[p2]

                domain_compatible = bool(sig1["domain"] & sig2["domain"])

                range_compatible = bool(sig1["range"] & sig2["range"])

                if not domain_compatible or not range_compatible:
                    continue

                score = float(mat[i, j])

                if score >= self.threshold:
                    union(canonical_map, predicate_freqs, p1, p2)

        # each predicates gets its correct final parent
        for predicate in canonical_map:
            canonical_map[predicate] = find(canonical_map, predicate)

        logger.info(
            "Predicate canonicalization=%s",
            canonical_map,
        )

        canonical_edges = [
            {
                **edge,
                "predicate": canonical_map[edge["predicate"]],
            }
            for edge in edges
        ]

        return canonical_edges
