# TruthfulRAG with GLiNER-Based Knowledge Graph Construction

This project aims to extend [**TruthfulRAG**](https://github.com/STAIR-BUPT/TruthfulRAG/tree/main) by replacing the LLM-based Knowledge Graph Construction with a faster pipeline, based on [**GLiNER**](https://github.com/urchade/GLiNER) (Generalist and Lightweight Model for Named Entity Recognition).

## 1. Project Overview and Research Questions

## 2. Methodology

## 3. Main Results

## 4. How to Reproduce

### Requirements

- Python 3.11
- CUDA-capable GPU with enough VRAM for `Qwen/Qwen2.5-7B-Instruct` and `knowledgator/gliner-relex-large-v0.5`
- Git and Conda or a different Python environment manager

All experiments were run on a single NVIDIA A100 (Colab and BwUniCluster3.0) or H100 (BwUniCluster3.0) GPU, based on availability.

## Repository Structure

```text
app/                         FastAPI demo service
datas/                       TimeQA, FaithEval, MuSiQue and SQuAD data
gliner_truthfulrag/
  adapter.py                 Converts GLiNER output to TruthfulRAGs KG format
  main.py                    Main experiment runner
  eval.py                    Markdown report generator
  kg_building.ipynb          KG construction walkthrough
  prompts/                   Predicate- and path-extraction prompts
  core/                      Environment management
  services/
    chunk_service.py         Token-based context chunking (GLiNER tokenizer)
    llm_service.py           Batched predicate extraction with Qwen
    ner_re_service.py        GLiNER entity and relation extraction
    kg_service.py            Orchestrator entrypoint for all services
    direct_path_service.py   Direct-path ablation
    helpers.py               Helpers for downloading, loading and offloading models
truthfulrag/                 Extended original TruthfulRAG implementation
outputs/                     Stored predictions, statistics and markdown reports
baselines.py                 Baseline experiment runner
baselines_eval.py            Baseline evaluation
run_experiment.ipynb         Example experiment walkthrough
```

## Citations

### TruthfulRAG

```bibtex
@article{liu2025truthfulrag,
  title={TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs},
  author={Liu, Shuyi and Shang, Yuming and Zhang, Xi},
  journal={arXiv preprint arXiv:2511.10375},
  year={2025}
}
```

### GLiNER

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
    title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author = "Zaratiana, Urchade and
      Tomeh, Nadi and
      Holat, Pierre and
      Charnois, Thierry",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    year = "2024",
    url = "https://aclanthology.org/2024.naacl-long.300",
    pages = "5364--5376",
}
```
