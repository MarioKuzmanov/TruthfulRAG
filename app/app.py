from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import router as main_router
from app.services.setup_service import SetupService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # we will use shared item across all router endpoints
    app.state.setup_service = SetupService()
    yield

usage_md = """
## How to use TruthfulRAG?

### Endpoints

- `setup`
    - selects a random `item_id` to examine the behaviour of a method
- `truthfulrag`
    - Original KG building steps
    - Integrated GLiNER KG building module
- `rag-triples`
    - RAG with provided extracted triples as additional context
- `rag`
    - simple RAG
- `llm`
    - LLM without context nor facts
    
- LLM is always `Qwen/Qwen2.5-7B-Instruct`
- GLiNER is always `knowledgator/gliner-relex-large-v0.5`
- Entropy filtering `threshold=3` (where applicable)
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
