# GLiNER KG Building

* drop-in replacement for the LLM KG building module in TruthfulRAG

---

## Architecture

```mermaid
flowchart TD
    Input["Data item"]

    subgraph KG["GLiNER KG Service"]
        Chunker["STEP: Chunker Service<br/>Token-based chunking<br/>GLiNER's tokenizer"]
        Predicates["STEP: Raw Predicate Extraction<br/>Main LLM is used"]
        Schema["Predefined entity schema<br/>(default)"]
        NERRE["STEP: NER and RE<br/>GLiNER uses entity and relation schemas<br/>FlashDeBERTa kernel backend"]
        Chunker -->|Text chunks| Predicates
        Chunker -->|Text chunks| NERRE
        Predicates -->|Relation schema| NERRE
        Schema --> NERRE
    end

    Integration["STEP: Integration<br/>Map nodes and relations<br/>Add native-supported TruthfulRAG backend and config"]
    RAG["TruthfulRAG<br/>Optionally with GLiNER `kg_backend`"]
    Evaluation["STEP: Evaluation<br/>Generate summary for an `experiment-id`"]
    Input --> Chunker
    NERRE -->|Nodes and relations| Integration
    Integration --> RAG
    RAG --> Evaluation
```

### Models

- main LLM: `Qwen/Qwen2.5-7B-Instruct`
- GLiNER: `knowledgator/gliner-relex-large-v0.5`

---

### Details

- sync design based on SLM + LLM approach

* `STEP: Chunker Service`
    * token-based chunking with overlap
    * tokens are given from GLiNER's tokenizer
    * conceptually same as the one in TruthfulRAG

* `STEP: Raw Predicate Extraction`
    * main LLM performs batched-raw predicate extraction from text chunks

* `STEP: NER and RE`
    * GLiNER performs NER and RE based on the pre-defined entities and relations schemas
    * Batched inference with full-precision weights and optimized `FlashDeBERTa` kernel backend

* `STEP: Integration`
    * `adapter.py` for integration into `TruthfulRAG`
    * native-support by customized `kg_backend` and `gliner_kg_service_config`

* `STEP: Evaluation`
    * `eval.py` for report generation with summaries for the experimental runs

---

### Structure

| Script                                   | Description                                            | 
|------------------------------------------|--------------------------------------------------------|
| `core/env_manager.py`                    | Environment management                                 | 
| `prompts/predicate_extraction_prompt.md` | Prompt for raw predicate extraction                    | 
| `adapter.py`                             | Integration entrypoint for `TruthfulRAG`               |
| `main.py`                                | Running experiments                                    |
| `services/eval.py`                       | Report generator for final summary of experiments      |
| `kg_building.ipynb`                      | Clean implementation of the KG Building method         |
| `deprecated/`                            | Implemented but currently unused services              |
| `services/chunk_service.py`              | Service for chunking by token size (GLiNER tokenizer)  |
| `services/llm_service.py`                | LLM-backend service for predicate extraction           |
| `services/ner_re_service.py`             | GLiNER-backend service for KG creation                 |
| `services/kg_service.py`                 | Orchestrator entrypoint for all services               |
| `services/helpers.py`                    | Helpers for downloading, loading and offloading models |

---

## Preliminary Results

- pilot study (first 15 items from `timeqa_2022_nota.json`)

### GLiNER

#### Quality

| num_items | exact_match |      acc |       f1 |
|----------:|------------:|---------:|---------:|
|        15 |    66.6667% | 66.6667% | 66.6667% |

#### Runtime

|    Total | KG Building | KG Retrieval | Entropy Filter | Prediction + Eval | Paths | Filtered Paths |
|---------:|------------:|-------------:|---------------:|------------------:|------:|---------------:|
| 159.73 s |     62.22 s |      46.66 s |         8.53 s |           42.32 s |    92 |             64 |

#### Runtime-average

| KG Building - per item | KG Retrieval - per item | Entropy Filter - per item | Prediction + Eval - per item | Paths - per item | Filtered Paths - per item |
|-----------------------:|------------------------:|--------------------------:|-----------------------------:|-----------------:|--------------------------:|
|                 4.15 s |                  3.11 s |                    0.57 s |                       2.82 s |             6.13 |                      4.27 |

---

### Qwen

#### Quality

| num_items | exact_match |      acc |       f1 |
|----------:|------------:|---------:|---------:|
|        15 |    80.0000% | 80.0000% | 80.0000% |

#### Runtime

|     Total | KG Building | KG Retrieval | Entropy Filter | Prediction + Eval | Paths | Filtered Paths |
|----------:|------------:|-------------:|---------------:|------------------:|------:|---------------:|
| 1754.06 s |   1648.37 s |      51.65 s |        12.21 s |           41.82 s |    93 |             79 |

#### Runtime-average

| KG Building - per item | KG Retrieval - per item | Entropy Filter - per item | Prediction + Eval - per item | Paths - per item | Filtered Paths - per item |
|-----------------------:|------------------------:|--------------------------:|-----------------------------:|-----------------:|--------------------------:|
|               109.89 s |                  3.44 s |                    0.81 s |                       2.79 s |             6.20 |                      5.27 |

### Comparison

* `TruthfulRAG-GLiNER-KG` is retaining 83.3% of the accuracy of the original `TruthfulRAG`
* `TruthfulRAG-GLiNER-KG` is ~ 11x faster than the original `TruthfulRAG`
* KG-construction is approximately 26.5x faster with `GLiNER-backend` than with `LLM-backend`

---

## Future Work

Potential directions for future work on the GLiNER approach include _runtime_ and _quality_ improvements:

### Runtime Improvements

- GLiNER integration with `https://pypi.org/project/disentangled-flash/` i.e. custom `flash-attn` kernel for the
  `DeBERTa` encoder (runtime improvement)

### Quality Improvements

- Deduplication and Linking steps
- GLiNER + GLiREL usage (separate NER and RE steps) -- threshold sensitivity experiments are needed.
- LLM-agentic refinement of the final produced graph
- Independent usage, not within the TruthfulRAG pipeline

---