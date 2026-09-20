# KG Building

* drop-in replacement for the LLM KG building module in TruthfulRAG

---

## Architecture

### Models

- main LLM: `Qwen/Qwen2.5-7B-Instruct`
- GLiNER: `knowledgator/gliner-relex-large-v0.5`

---

### Details

- sync design without introducing additional hardware dependencies

* `STEP: Chunker Service`
    * token-based chunking with overlap
    * tokens are given from GLiNER's tokenizer
    * conceptually same as the one in TruthfulRAG

* `STEP: Raw Predicate Extraction`
    * main LLM performs raw predicate extraction from text chunks

* `STEP: NER and RE`
    * GLiNER performs NER and RE based on the pre-defined entities and relations schemas
    * Batched inference with full-precision weights and optimized `FlashDeBERTa` backend

* `STEP: Integration`
    * `adapter.py` for integration into `TruthfulRAG`

---

### Structure

| Script                                   | Description                                    | 
|------------------------------------------|------------------------------------------------|
| `core/env_manager.py`                    | Environment management                         | 
| `prompts/predicate_extraction_prompt.md` | Prompt for raw predicate extraction            | 
| `adapter.py`                             | Integration entrypoint for `TruthfulRAG`       |
| `main.py`                                | Running experiments                            |
| `kg_building.ipynb`                      | Clean implementation of the KG Building method |
| `deprecated/`                            | Implemented but currently unused services      |
| `services/`                              |                                                |

---