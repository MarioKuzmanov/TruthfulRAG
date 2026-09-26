from fastapi import APIRouter, Response, Depends, HTTPException
from app.services.setup_service import get_setup_service, SetupService
import textwrap

router = APIRouter()


@router.post("/select-shared-item/")
def select_random_item_for_all_methods(item_service: SetupService = Depends(get_setup_service)):
    item_meta = item_service.get_random_item()
    item_id = item_service.curr_item_id

    if item_meta is None:
        raise HTTPException(status_code=404, detail="Item not found")

    choices = "\n".join(choice for choice in item_meta["choices"])

    context = "\n\n".join(
        textwrap.fill(paragraph, width=100)
        for paragraph in item_meta["context"].split("\n\n")
    )

    content = (f"## ITEM_ID\n\n"
               f"{item_id}\n\n"
               f"---\n\n"
               f"## QUESTION\n\n"
               f"{item_meta['question']}\n\n"
               f"---\n\n"
               f"## CHOICES\n\n"
               f"{choices}\n\n"
               f"---\n\n"
               f"## GOLD ANSWER\n\n"
               f"{item_meta['gold_answer']}\n\n"
               f"---\n\n"
               f"## CONTEXT\n\n"
               f"{context}\n\n")

    response = Response(
        content=content,
        status_code=200,
        media_type="text/markdown"
    )

    return response
