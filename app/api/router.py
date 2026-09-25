from fastapi import APIRouter
from app.api.endpoints.endpoints_truthfulrag_original.truthfulrag_original import router as original_router
from app.api.endpoints.endpoints_truthfulrag_gliner.truthfulrag_gliner import router as gliner_router
from app.api.endpoints.endpoints_rag_triples.rag_triples import router as rag_triples_router
from app.api.endpoints.endpoints_rag.rag import router as rag_router
from app.api.endpoints.endpoints_llm.llm import router as llm_router

router = APIRouter()

router.include_router(original_router, prefix="/truthfulrag-original", tags=["truthfulrag-original"])
router.include_router(gliner_router, prefix="/truthfulrag-gliner", tags=["truthfulrag-gliner"])
router.include_router(rag_triples_router, prefix="/rag-triples", tags=["rag-triples"])
router.include_router(rag_router, prefix="/rag", tags=["rag"])
router.include_router(llm_router, prefix="/llm", tags=["llm"])
