from fastapi import APIRouter, Depends, Response
from app.services.prompt_service import AbstractPromptService, get_prompt_service

router = APIRouter()


@router.get("/prompt-direct-path-extraction")
def prompt_original_extraction(prompt_service: AbstractPromptService = Depends(get_prompt_service)):
    content = prompt_service.prompt_direct_path_extraction()

    return Response(content=content, status_code=200, media_type="text/markdown")
