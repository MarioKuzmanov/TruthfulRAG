from fastapi import APIRouter, Depends, Response, HTTPException
from app.services.rag_triples_service import RAGTriplesService, get_rag_triples_service
from app.services.setup_service import get_selected_item_id

import textwrap

router = APIRouter()


@router.get("/view-extracted-paths")
def view_extracted_paths(
        item_id: str = Depends(get_selected_item_id),
        rag_triples_service: RAGTriplesService = Depends(get_rag_triples_service)):
    paths = rag_triples_service.get_paths(item_id)

    if paths is None:
        raise HTTPException(status_code=404, detail="Item not found")

    all_paths = "\n\n".join(
        textwrap.fill(path, width=100)
        for path in paths["extracted_paths"]
    )

    filtered_paths = "\n\n".join(
        f"PATH:\n\n{path['element']}\n\nENTROPY_DELTA:\n\n{path['delta']}"
        for path in paths["filtered_paths"]
    )

    response = (f"## EXTRACTED_PATHS\n\n"
                f"{all_paths}\n\n"
                f"---\n\n"
                f"## FILTERED_PATHS\n\n"
                f"{filtered_paths}\n\n---\n\n")

    return Response(content=response, status_code=200, media_type="text/markdown")


@router.get("/response-wo-context")
def response_without_context(item_id: str = Depends(get_selected_item_id),
                             rag_triples_service: RAGTriplesService = Depends(get_rag_triples_service)):
    response = rag_triples_service.get_response_wo_context(item_id)
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


@router.get("/response-with-context")
def response_with_context(item_id: str = Depends(get_selected_item_id),
                          rag_triples_service: RAGTriplesService = Depends(get_rag_triples_service)):
    response = rag_triples_service.get_response_with_context(item_id)
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
