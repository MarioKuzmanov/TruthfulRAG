from fastapi import APIRouter, Response, Depends, HTTPException
from app.services.setup_service import get_selected_item_id
from app.services.llm_service import get_llm_service, LLMService

router = APIRouter()


@router.get("/response-wo-cot/")
def response_without_cot(item_id: str = Depends(get_selected_item_id),
                         llm_service: LLMService = Depends(get_llm_service)):
    response = llm_service.response_without_cot(item_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Item not found")

    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/response-cot/")
def response_with_cot(item_id: str = Depends(get_selected_item_id), llm_service: LLMService = Depends(get_llm_service)):
    response = llm_service.response_with_cot(item_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Item not found")

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

