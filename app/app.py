from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import router as main_router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up TruthfulRAG")

    yield

    logger.info("Cleaning up TruthfulRAG")


usage_md = """
## How to use TruthfulRAG?

### Endpoints
- `truthfulrag-original`
- `truthfulrag-gliner`
- `rag-triples`
- `rag`
- `llm`
"""
app = FastAPI(title="TruthfulRAG Demo Service",
              version="0.1.0",
              openapi_url=f"/api/openapi.json",
              lifespan=lifespan,
              description=usage_md)
app.include_router(main_router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "alive"}
