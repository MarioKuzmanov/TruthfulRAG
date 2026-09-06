# GLiNER text2kg

* drop-in replacement for the LLM KG building module in TruthfulRAG

## Architecture

### Details

- sync design
- parallel execution across input items is limited for a single GPU

* `STEP: Chunking`
    * token-based chunking with overlap
    * tokens are given from GLiNER's tokenizer
    * conceptually same as the one in TruthfulRAG
* `STEP: NER`
    * GLiNER-based model is used for batch inference on a single GPU
    * Loaded with optimized FlashDeBERTa backend, and in FP16 precision
* `STEP: Dedup NER`
    * Fast deduplication to determine per chunk canonical nodes/entities
    * Used for RE
* `STEP: RE`
    * Based on the backend - `openai`, `qwen` or `mistral`. Prompt for `openai` taken from `KGGen`
    * `openai`
        * parallelized RE across chunks with API calls
* `STEP: Dedup RE`
    * Fast deduplication to determine per chunk canonical edges
* `KG Ensemble`
    * KG from all chunks with semantic dedup

This plan is in progress and subject to changes

## Steps

* implement drop-in replacement
* integrate into TruthfulRAG
* benchmark KG building
* benchmark vs RAG and TruthfulRAG

---