import asyncio
import json

from truthfulrag.prompts import GRAPH_FIELD_SEP
from truthfulrag.utils import compute_mdhash_id


def _entity_description(name: str, entity_type: str) -> str:
    # we need generic descriptions without LLM calls for compatibility with TruthfulRAG
    return f'{name} is an entity of type {entity_type}'


def adapt_kg_records(sample: dict, nodes: dict, relations: list[dict]):
    source_id = str(sample["id"])

    graph_nodes = {}
    for node, node_metadata in nodes.items():
        # multi-labels are resolved by here
        node_metadata = node_metadata[0]
        entity_type = node_metadata["entity_type"]
        chunk_id = node_metadata["chunk_id"]
        confidence = node_metadata["confidence"]

        graph_nodes[node] = {
            "entity_type": entity_type,
            "description": _entity_description(node, entity_type),
            "source_id": source_id,
            "chunk_id": chunk_id,
            "confidence": confidence
        }

    graph_edges = {}
    for rel in relations:
        head_node_meta, tail_node_meta = rel["head"], rel["tail"]
        rel_type, confidence = rel["relation"], rel["score"]

        pair = (head_node_meta["entity_name"].upper(), tail_node_meta["entity_name"].upper())
        chunk_ids = sorted([head_node_meta["chunk_id"], tail_node_meta["chunk_id"]])

        graph_edges[pair] = {
            "description": f"{head_node_meta['entity_name'].upper()} {rel_type.replace('_', ' ')} {tail_node_meta['entity_name'].upper()}",
            "keywords": GRAPH_FIELD_SEP.join([head_node_meta["entity_type"], rel_type, tail_node_meta["entity_type"]]),
            "source_id": source_id,
            "chunk_ids": GRAPH_FIELD_SEP.join(chunk_ids),
            "weight": confidence,
        }

    return graph_nodes, graph_edges


async def make_gliner_kg(sample, kg_service, knowledge_graph_inst, entities_vdb, entity_name_vdb,
                         relationships_vdb):
    # integrate with the implemented service
    nodes, relations = await asyncio.to_thread(kg_service.build_kg_step, item=sample)
    graph_nodes, graph_edges = adapt_kg_records(sample, nodes, relations)

    # populate databases -> knowledge graph instance
    for name, data in graph_nodes.items():
        await knowledge_graph_inst.upsert_node(name, node_data=data)
    for (source, target), data in graph_edges.items():
        await knowledge_graph_inst.upsert_edge(source, target, edge_data=data)

    if entities_vdb is not None and graph_nodes:
        await entities_vdb.upsert({
            compute_mdhash_id(name, prefix="ent-"): {
                "entity_name": name,
                "content": name + ": " + data["description"],
            }
            for name, data in graph_nodes.items()
        })
    if entity_name_vdb is not None and graph_nodes:
        await entity_name_vdb.upsert({
            compute_mdhash_id(name, prefix="Ename-"): {
                "entity_name": name,
                "content": name,
            }
            for name in graph_nodes
        })
    if relationships_vdb is not None and graph_edges:
        await relationships_vdb.upsert({
            compute_mdhash_id(json.dumps([source, target], ensure_ascii=False), prefix="rel-"): {
                "src_id": source,
                "tgt_id": target,
                "content": " ".join([data["keywords"], source, target, data["description"]]),
            }
            for (source, target), data in graph_edges.items()
        })

    return knowledge_graph_inst
