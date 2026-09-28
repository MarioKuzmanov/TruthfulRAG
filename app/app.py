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

### What should you know about the service?

- the service is a lightweight demo of the behavior of all implemented methods
- it works with a pre-defined subset of items (refer to `gliner_truthfulrag/main.py` for full experimental runs)
- it is recommended for a single-user

### How to use the service?

**Recommended workflow**

1. Select a random `item_id`

2. Inspect the internals of all available methods

### Endpoints

- `setup`
    - select a random `item_id` to examine the behaviour of a method

- `prompts`
    - inspect the main prompts used for the methods
    
- `llm`
    - LLM-only with and without CoT 
    
- `rag`
    - simple RAG with and without CoT

- `rag-triples`
    - RAG with provided extracted triples as additional context for QA
    
- `truthfulrag-gliner`
    - with the integrated GLiNER-based pipeline
    - responses with and without context for QA
    
- `truthfulrag-original`
    - with the original KG building pipeline
    - responses with and without context for QA

- LLM = `Qwen/Qwen2.5-7B-Instruct`
- GLiNER = `knowledgator/gliner-relex-large-v0.5`
- Entropy filtering threshold = 3 (where applicable)

---

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
