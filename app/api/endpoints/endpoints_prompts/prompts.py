from fastapi import APIRouter, Response, Depends
from app.services.prompt_service import PromptService, get_prompt_service

router = APIRouter()


@router.get("/wo-cot-wo-facts")
def prompt_without_cot_without_facts(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_wo_cot_wo_facts()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/cot-wo-facts")
def prompt_cot_without_facts(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_cot_wo_facts()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/cot-facts")
def prompt_cot_with_facts(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_cot_facts()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/cot-facts-wo-context")
def prompt_cot_with_facts_without_context(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_cot_facts_wo_context()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/rag-triples-direct-path-extraction")
def prompt_rag_triples_direct_path_extraction(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_direct_path_extraction()

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/truthfulrag-gliner-raw-predicate-extraction")
def prompt_truthfulrag_gliner_raw_predicate_extraction(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_raw_predicate_extraction()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/truthfulrag-original-entity-extraction")
def prompt_original_extraction(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_original_entity_extraction()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/truthfulrag-original-query-to-keywords")
def prompt_query_to_keywords(prompt_service: PromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_original_query2kwd()
    return Response(content=content, status_code=200, media_type="text/markdown")

# @router.get("/wo-facts-wo-cot/")
# def get_prompt_wo_cot(prompt_service: PromptService = Depends(get_prompt_service)):
#     content = prompt_service.prompt_llm_wo_cot()
#     return Response(content=content, status_code=200, media_type="text/markdown")
#
#
# @router.get("/llm-cot/")
# def get_prompt_cot(prompt_service: PromptService = Depends(get_prompt_service)):
#     content = prompt_service.prompt_llm_with_cot()
#     return Response(content=content, status_code=200, media_type="text/markdown")
#
#
# @router.get("/rag-wo-cot/")
# def get_prompt_wo_cot(prompt_service: PromptService = Depends(get_prompt_service)):
#     content = prompt_service.prompt_rag_wo_cot()
#     return Response(content=content, status_code=200, media_type="text/markdown")
#
#
# @router.get("/rag-cot/")
# def get_prompt_cot(prompt_service: PromptService = Depends(get_prompt_service)):
#     content = prompt_service.prompt_rag_with_cot()
#     return Response(content=content, status_code=200, media_type="text/markdown")
