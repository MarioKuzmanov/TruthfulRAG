from fastapi import APIRouter, Response

router = APIRouter()


@router.post("/{item}")
def run_simulate_async_pipeline(item: str):
    return Response(content=item, status_code=200)
