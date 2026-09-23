from typing import List, Dict, Literal
import re
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import json
import asyncio
from collections import Counter
from collections import defaultdict
from .prompts import PromptGenerator, GRAPH_FIELD_SEP, PROMPTS
import heapq
from .llm import LLMBackend
from util import logger
import json_repair

from .utils import (
    split_by_sentence,
    split_string_by_multi_markers,
    logger,
    clean_str,
    edge_vote_path,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    compute_mdhash_id,
    pack_user_ass_to_openai_messages,
    cal_path_score_list,
    remove_subpaths
)
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    QueryParam,
)

EntropyFilterMethod = Literal["legacy", "paper"]


def chunking_by_token_size(
        item: Dict, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(item['context'], model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start: start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return {str(item['id']): results}


async def generate_knowledge_graph(
        filtered_chunks: List[Dict],
        similarity_model: str,
        backend_type: str,
        model_name: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        entity_name_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        **backend_config
) -> List[Dict]:
    """
    Generate knowledge graph for each item in the dataset

    Args:
        filtered_chunks: filtered chunks for knowledge graph generation
        similarity_model: similarity model for calculating similarity between entities
        backend_type: backend type for generation
        model_name: model name for generation
        knowledge_graph_inst: instance of knowledge graph storage
        entities_vdb: instance of entity vector storage
        entity_name_vdb: instance of entity name vector storage
        relationships_vdb: instance of relationship vector storage
        backend_config: Generation parameters to override defaults

    Returns:
        List of dictionaries containing knowledge graphs for each item
    """
    similarity_model = SentenceTransformer(similarity_model)
    llm_backend = LLMBackend(
        backend_type=backend_type,
        model_name=model_name,
        **backend_config
    )
    default_sampling_params = {
        'max_tokens': 1000,
        'top_p': 1.0
    } if backend_type == 'openai' else {
        'max_new_tokens': 1000,
        'do_sample': False
    }

    async def _process_single_chunk(single_chunk):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = single_chunk[0]
        chunk_dp = single_chunk[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await llm_backend.single_generate(
            prompt=hint_prompt,
            system_prompt=None,
            **merged_params
        )
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        entity_extract_max_gleaning = 1
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await llm_backend.single_generate(
                prompt=continue_prompt,
                system_prompt=None,
                history_messages=history,
                **merged_params
            )
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await llm_backend.single_generate(
                prompt=if_loop_prompt,
                system_prompt=None,
                history_messages=history,
                **merged_params
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
            ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    merged_params = {**default_sampling_params, **backend_config}
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    key, value = next(iter(filtered_chunks.items()))
    ordered_chunks = [(key, v) for v in value]
    already_processed = 0
    already_entities = 0
    already_relations = 0
    results = await asyncio.gather(
        *[_process_single_chunk(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entities_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + ": " + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entities_vdb.upsert(data_for_vdb)
    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                           + " " + dp["src_id"]
                           + " " + dp["tgt_id"]
                           + " " + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def retrieve_knowledge_graph(
        sample: Dict,
        similarity_model: str,
        backend_type: str,
        model_name: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        entity_name_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        **backend_config
) -> List[Dict]:
    """
    Retrieve knowledge graph for each item in the dataset

    Args:
        sample: Input sample
        similarity_model: similarity model for calculating similarity between entities
        backend_type: backend type for generation
        model_name: model name for generation
        knowledge_graph_inst: instance of knowledge graph storage
        entities_vdb: instance of entity vector storage
        entity_name_vdb: instance of entity name vector storage
        relationships_vdb: instance of relationship vector storage
        backend_config: Generation parameters to override defaults

    Returns:
        List of dictionaries containing retrieved knowledge graphs for each item
    """
    similarity_model = SentenceTransformer(similarity_model)
    llm_backend = LLMBackend(
        backend_type=backend_type,
        model_name=model_name,
        **backend_config
    )
    default_sampling_params = {
        'max_tokens': 1000,
        'top_p': 1.0
    } if backend_type == 'openai' else {
        'max_new_tokens': 1000,
        'do_sample': False
    }
    merged_params = {**default_sampling_params, **backend_config}
    kw_prompt_temp = PROMPTS["truthfulrag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=sample['question'], TYPE_POOL=TYPE_POOL)
    result = await llm_backend.single_generate(
        prompt=kw_prompt,
        system_prompt=None,
        **merged_params
    )

    try:
        keywords_data = json_repair.loads(result)
        print(f"keywords_data: {keywords_data}\n\n")
        type_keywords = keywords_data.get("answer_type_keywords", [])
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # Handle parsing error
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    query_param = QueryParam()
    context = await _build_query_context(
        entities_from_query,
        type_keywords,
        sample['question'],
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        query_param,
    )
    # if not context:
    #     context = split_by_sentence(sample['context'])
    return context


async def entropy_filter(
        sample: Dict,
        elements: List[Dict],
        backend_type: str,
        model_name: str,
        top_k: int = 10,
        threshold: float = 1,
        entropy_filter_method: EntropyFilterMethod = "legacy",
        **backend_config
) -> List[Dict]:
    """
    Apply entropy filtering to the retrieved knowledge graph elements

    Args:
        sample: Input sample
        elements: List of knowledge graph elements
        backend_type: Backend type for generation
        model_name: Model name for generation
        top_k: Maximum number of elements to keep with the legacy method
        threshold: Method-specific entropy threshold
        entropy_filter_method: Entropy filtering implementation to use
        backend_config: Generation parameters to override defaults

    Returns:
        Filtered list of elements
    """
    if entropy_filter_method not in {"legacy", "paper"}:
        raise ValueError("entropy_filter_method must be 'legacy' or 'paper'")

    # Log base is different for legacy and paper methods
    entropy_log_base = "natural" if entropy_filter_method == "legacy" else "base2"

    llm_backend = LLMBackend(
        backend_type=backend_type,
        model_name=model_name,
        **backend_config
    )

    default_sampling_params = {
        'max_tokens': len(encode_string_by_tiktoken(sample["answer"])),
        'top_p': 1.0
    } if backend_type == 'openai' else {
        'max_new_tokens': len(encode_string_by_tiktoken(sample["answer"])),
        'do_sample': False
    }

    prompt_generator = PromptGenerator(
        llm_type=backend_type,
        task="qa"
    )

    merged_params = {**default_sampling_params, **backend_config}

    # Step 1: Compute baseline entropy without any facts
    baseline_prompt = prompt_generator.generate_qa_prompt(
        context="",
        question=sample['question'],
        options=sample.get('choices'),
        facts=""
    )

    baseline_entropy = await llm_backend.cal_entroy(
        prompt=baseline_prompt,
        system_prompt=prompt_generator.system_prompt,
        answer=sample["answer"],
        entropy_log_base=entropy_log_base,
        **merged_params
    )

    print(f"Baseline entropy: {baseline_entropy}\n\n")

    # Step 2: Compute entropy for each element and calculate entropy reduction
    entropy_deltas = []
    entropy_deltas_dict = defaultdict(float)
    for element in elements:
        prompt_with_fact = prompt_generator.generate_qa_prompt(
            context="",
            question=sample['question'],
            options=sample.get('choices'),
            facts=element
        )

        entropy_with_fact = await llm_backend.cal_entroy(
            prompt=prompt_with_fact,
            system_prompt=prompt_generator.system_prompt,
            answer=sample["answer"],
            entropy_log_base=entropy_log_base,
            **merged_params
        )

        print(f"entropy_with_fact: {entropy_with_fact}\n\n")
        entropy_deltas_dict[element] = entropy_with_fact
        entropy_delta = entropy_with_fact - baseline_entropy

        # Entropy filtering based on paper method
        if entropy_filter_method == "paper":
            if entropy_delta > threshold:
                entropy_deltas.append({"element": element})
        # Entropy filtering based on legacy repo method
        elif entropy_delta >= 0.0:
            entropy_deltas.append({
                "element": element,
                "delta": entropy_delta
            })
        elif entropy_with_fact != 0.0 and baseline_entropy / entropy_with_fact <= 10 ** threshold:
            entropy_deltas.append({
                "element": element,
                "delta": entropy_delta
            })
    # Return here for the paper method
    # If the list is empty an empty list will be returned (different from the legacy repo method)
    if entropy_filter_method == "paper":
        return [item["element"] for item in entropy_deltas]
    
    # Handle the case when no elements passed the entropy filter, 
    # resulting in selecting the element with the maximum entropy delta
    if not entropy_deltas and entropy_deltas_dict:
        max_entropy_delta = max(entropy_deltas_dict.values())
        entropy_deltas = [
            {
                "element": k,
                "delta": v
            }
            for k, v in entropy_deltas_dict.items() if v == max_entropy_delta
        ]

    top_filtered = heapq.nlargest(top_k, entropy_deltas, key=lambda x: x["delta"])
    return [item["element"] for item in top_filtered]


async def predict_answer(
        dataset: Dataset,
        elements: List[Dict],
        backend_type: str,
        model_name: str,
        generation_type: str = "cot",
        generation_context: str = "original",
        **backend_config
) -> Dict[str, str]:
    """
    Predict answers using chain-of-thought reasoning

    Args:
        dataset: Input dataset
        elements: Factual knowledge for each item
        backend_type: Backend type for generation
        model_name: Model name for generation
        generation_type: Type of generation to use
        generation_context: Defines whether to include the original context.
        backend_config: Generation parameters to override defaults

    Returns:
        Dictionary of predictions keyed by item ID
    """
    if generation_context not in {"original", "none"}:
        raise ValueError("generation_context must be 'original' or 'none'")

    # Initialize LLM backend
    llm_backend = LLMBackend(
        backend_type=backend_type,
        model_name=model_name,
        **backend_config
    )

    # Initialize prompt generators
    prompt_generator_qa_cot = PromptGenerator(
        llm_type=backend_type,
        task="qa-cot"
    )
    prompt_generator_qa = PromptGenerator(
        llm_type=backend_type,
        task="qa"
    )

    # Default sampling parameters
    default_sampling_params = {
        'max_tokens': 1000,
        'top_p': 1.0
    } if backend_type == 'openai' else {
        'max_new_tokens': 1000,
        'do_sample': False
    }
    # Generate prompts
    prompts = []
    for item in dataset:
        element_list = []
        for e in elements:
            if e['id'] == item['id']:
                element_list.extend(e['element'])
        elements_str = '\n\n'.join(element_list)
        print(f"elements_str: {elements_str}\n\n")
        
        include_context = generation_context == "original"  # False if "none"

        if generation_type == "cot":
            prompts.append(
                prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                    context=item.get('context', '') if include_context else '',
                    question=item['question'],
                    options=item.get('choices'),
                    facts=elements_str,
                    include_context=include_context
                )
            )
        else:
            prompts.append(
                prompt_generator_qa.generate_qa_prompt(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=elements_str
                )
            )

    # Generate responses
    merged_params = {**default_sampling_params, **backend_config}
    if generation_type == "cot":
        results = await llm_backend.generate(
            prompts=prompts,
            system_prompt=prompt_generator_qa_cot.system_prompt,
            **merged_params
        )
    else:
        results = await llm_backend.generate(
            prompts=prompts,
            system_prompt=prompt_generator_qa.system_prompt,
            **merged_params
        )

    # Return predictions
    return {item['id']: res for item, res in zip(dataset, results)}


async def predict_answer_wo_facts(
        dataset: Dataset,
        backend_type: str,
        model_name: str,
        withrag: bool = False,
        generation_type: str = "cot",
        **backend_config
) -> Dict[str, str]:
    """
    Predict answers using chain-of-thought reasoning

    Args:
        dataset: Input dataset
        backend_type: Backend type for generation
        model_name: Model name for generation
        withrag: Whether to use RAG or not
        generation_type: Type of generation to use
        backend_config: Generation parameters to override defaults

    Returns:
        Dictionary of predictions keyed by item ID
    """
    # Initialize LLM backend
    llm_backend = LLMBackend(
        backend_type=backend_type,
        model_name=model_name,
        **backend_config
    )

    # Initialize prompt generators
    prompt_generator_qa_cot = PromptGenerator(
        llm_type=backend_type,
        task="qa-cot"
    )
    prompt_generator_qa = PromptGenerator(
        llm_type=backend_type,
        task="qa"
    )

    # Default sampling parameters
    default_sampling_params = {
        'max_tokens': 1000,
        'top_p': 1.0
    } if backend_type == 'openai' else {
        'max_new_tokens': 1000,
        'do_sample': False
    }
    # Generate prompts
    prompts = []
    for item in dataset:
        if withrag:
            if generation_type == "cot":
                prompts.append(
                    prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                        facts=""
                    )
                )
            else:
                prompts.append(
                    prompt_generator_qa.generate_qa_prompt(
                        context=item.get('context', ''),
                        question=item['question'],
                        options=item.get('choices'),
                        facts=""
                    )
                )
        else:
            if generation_type == "cot":
                prompts.append(
                    prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                        context="",
                        question=item['question'],
                        options=item.get('choices'),
                        facts=""
                    )
                )
            else:
                prompts.append(
                    prompt_generator_qa.generate_qa_prompt(
                        context="",
                        question=item['question'],
                        options=item.get('choices'),
                        facts=""
                    )
                )

    # Generate responses
    merged_params = {**default_sampling_params, **backend_config}
    if generation_type == "cot":
        results = await llm_backend.generate(
            prompts=prompts,
            system_prompt=prompt_generator_qa_cot.system_prompt,
            **merged_params
        )
    else:
        results = await llm_backend.generate(
            prompts=prompts,
            system_prompt=prompt_generator_qa.system_prompt,
            **merged_params
        )

    # Return predictions
    return {item['id']: res for item, res in zip(dataset, results)}


async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)

    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def _build_query_context(
        ent_from_query,
        type_keywords,
        originalquery,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        entity_name_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        query_param: QueryParam,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )
    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    pairs_append = remove_subpaths(pairs_append)
    elements = []
    edge_set = set()
    for edges in pairs_append.values():
        for edge in edges:
            edge_set.add(edge)
    edges_list = list(edge_set)
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    edges_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_edge(edge[0], edge[1])
            for edge in edges_list
        ]
    )
    nodes_description = {entity_name: node_data["description"] for entity_name, node_data in
                         zip(scored_edged_reasoning_path.keys(), node_datas)}
    edges_description = {(edge[0], edge[1]): edges_data["description"] for edge, edges_data in
                         zip(edges_list, edges_datas)}

    for k, v in pairs_append.items():
        path_str = '->'.join(k)
        node_str = '\n'.join(
            list((str(e) + ": " + nodes_description[e]) for e in k if e in scored_edged_reasoning_path.keys()))
        edge_str = '\n'.join(list((str(e[0]) + '->' + str(e[1]) + ": " + edges_description[e]) for e in v))
        element_str = f"Path:\n{path_str}\nNodes:\n{node_str}\nEdges:\n{edge_str}"
        elements.append(element_str)
    return elements
