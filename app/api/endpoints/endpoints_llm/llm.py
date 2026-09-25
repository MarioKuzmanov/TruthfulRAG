from fastapi import APIRouter, Response, Depends
from app.services.prompt_service import AbstractPromptService, get_prompt_service
from app.services.registry_service import AbstractRegistryService, get_registry_service

router = APIRouter()


@router.post("/{item}")
def run_llm(item: str):
    return Response(content=item, status_code=200)


@router.get("/cot/{item_id}")
def response_cot(item_id: str):
    return Response(content=item_id, status_code=200)


@router.get("/cot-prompt/")
def get_prompt_cot(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_llm_with_cot()
    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/wo-cot/{item_id}")
def response_wo_cot(item_id: str):
    return Response(content=item_id, status_code=200)


@router.get("/wo-cot-prompt/")
def get_prompt_wo_cot(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_llm_wo_cot()
    return Response(content=content, status_code=200, media_type="text/markdown")
