from fastapi import APIRouter, Depends, Response
from app.services.setup_service import get_selected_item_id
from app.services.rag_service import RAGService, get_rag_service
from app.services.prompt_service import AbstractPromptService, get_prompt_service

router = APIRouter()


@router.get("/response-wo-cot")
def response_wo_cot(item_id: str = Depends(get_selected_item_id), rag_service: RAGService = Depends(get_rag_service)):
    response = rag_service.response_without_cot(item_id)

    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/response-cot")
def response_with_cot(item_id: str = Depends(get_selected_item_id), rag_service: RAGService = Depends(get_rag_service)):
    response = rag_service.response_with_cot(item_id)
    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## REASONING\n\n"
               f"{response['reasoning']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/prompt-wo-cot/")
def get_prompt_wo_cot(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_rag_wo_cot()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/prompt-cot/")
def get_prompt_cot(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_rag_with_cot()
    return Response(content=content, status_code=200, media_type="text/markdown")
