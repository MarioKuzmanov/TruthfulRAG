from fastapi import APIRouter
from app.api.endpoints.endpoints_setup.setup import router as setup_router
from app.api.endpoints.endpoints_truthfulrag.truthfulrag import router as truthfulrag_router
from app.api.endpoints.endpoints_rag_triples.rag_triples import router as rag_triples_router
from app.api.endpoints.endpoints_rag.rag import router as rag_router
from app.api.endpoints.endpoints_llm.llm import router as llm_router

router = APIRouter()

router.include_router(setup_router, prefix="/setup", tags=["setup"])
router.include_router(truthfulrag_router, prefix="/truthfulrag", tags=["truthfulrag"])
router.include_router(rag_triples_router, prefix="/rag-triples", tags=["rag-triples"])
router.include_router(rag_router, prefix="/rag", tags=["rag"])
router.include_router(llm_router, prefix="/llm", tags=["llm"])
