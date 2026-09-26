from fastapi import APIRouter, Depends, Response
from app.services.setup_service import get_selected_item_id
from app.services.truthfulrag_service import TruthfulRAGService, get_truthfulrag_service
from app.services.prompt_service import AbstractPromptService, get_prompt_service

router = APIRouter()


@router.get("/original-filtered-chunks")
def original_filtered_chunks(item_id: str = Depends(get_selected_item_id),
                             truthfulrag_service: TruthfulRAGService = Depends(get_truthfulrag_service)):
    response = truthfulrag_service.original_filtered_chunks()

    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/prompt-original-extraction")
def prompt_original_extraction(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_truthfulrag_llm_kg()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/prompt-query-to-keywords")
def prompt_query_to_keywords(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_truthfulrag_query2kwds()
    return Response(content=content, status_code=200, media_type="text/markdown")
