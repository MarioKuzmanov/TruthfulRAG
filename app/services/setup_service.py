import random
from fastapi import Request, Depends, HTTPException
from app.core.registry import REGISTERED_ITEMS_IDS, REGISTERED_ITEMS


class SetupService:
    def __init__(self):
        self.curr_item_id = None

    def get_random_item(self) -> dict:

        # item to demo is chosen arbitrary (from 5 possible choices)
        item_id = random.choice(REGISTERED_ITEMS_IDS)
        self.curr_item_id = str(item_id)

        for registered in REGISTERED_ITEMS:
            if registered["id"] == item_id:
                return {"question": registered['question'], "choices": registered["choices"],
                        "context": registered["context"], "gold_answer": registered['answer']}
        return None


def get_setup_service(request: Request) -> SetupService:
    return request.app.state.setup_service


def get_selected_item_id(setup_service: SetupService = Depends(get_setup_service)) -> str:
    """FastAPI callable dependency SHARED ACROSS ROUTERS"""
    item_id = setup_service.curr_item_id

    if item_id is None:
        raise HTTPException(status_code=404, detail="Item not found")

    return item_id
