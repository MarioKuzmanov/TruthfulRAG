# TruthfulRAG with GLiNER-Based Knowledge Graph Construction

This project aims to extend [**TruthfulRAG**](https://github.com/STAIR-BUPT/TruthfulRAG/tree/main) by replacing the LLM-based Knowledge Graph Construction with a faster pipeline, based on [**GLiNER**](https://github.com/urchade/GLiNER) (Generalist and Lightweight Model for Named Entity Recognition).

## 1. Project Overview and Research Questions

Retrieval-Augmented Generation (RAG) provides Large Language Models (LLMs) with external source data before generation. Nonetheless, those recovered sources may contain knowledge which conflicts with the knowledge the LLM learns during training. The paper [TruthfulRAG](https://arxiv.org/abs/2511.10375) presents a framework to resolving these factual-level conflicts. The author's approach entails the construction of Knowledge Graphs (KGs) from subject-predicate-object triples extracted from the external source, two-hop graph traversal for reasoning path retrieval, path relevancy ranking based on the query and entropy-based path filtering before final generation.

In the original implementation, large parts of the KG construction step (i.e., entity extraction, relation-label generation, and triple extraction) are performed by an LLM. This requires multiple LLM inference calls, making the pipeline resource- and time-intensive. Additionally, we identified several inconsistencies:

- **Entropy-filtering discrepancy:** The formulas described in Equations (9) and (11) in the paper differ from the formula implemented in the official repository.
- **Final-generation discrepancy:** Equation (12) in the paper does not mention the entire source context, although the implementation provides it alongside the reasoning paths during final generation.

Our main contribution is the replacement of the costly LLM-based KG backend with a faster backend based on GLiNER-relex, a 500M-parameter specialized model for zero-shot Named Entity Recognition and Relationship Extraction. The remainder of the TruthfulRAG pipeline, which includes graph retrieval, entropy-based filtering, final generation and evaluation, remains unchanged, allowing a fair comparison between both KG construction methods.

This project primarily aims to address the following questions:

1. **Reproduction**: Can we replicate the results reported in the original paper?
2. **Efficiency and Quality**: How does our GLiNER-based pipeline compare to the TruthfulRAG pipeline, in regards to answer quality and runtime?
3. **Role of Reasoning Paths**: To what extent does the inclusion of reasoning paths affect answer quality, with and without appending the entire original context.

Additionally, we partly address further questions. These however are not the main focus of this project and can be further investigated in the future:

4. **Entropy-filtering consistency**: How does the entropy-filtering method described in the paper compare with the method currently implemented in the official repository?
5. **Necessity of KG Construction**: Can query-aware reasoning paths be extracted directly from the source context while preserving answer quality and reducing runtime?
6. **Practical Deployability**: Can the proposed pipeline be integrated into TruthfulRAG as a deployable service?

> **Disclaimer**: The original paper reports experiments on the datasets FaithEval, MuSiQue, RealtimeQA and SQuAD across the three models `GPT-4o-mini`, `Qwen2.5-7B-Instruct` and `Mistral-7B-Instruct`. Due to time and resource constraints, we chose to run all experiments on `Qwen2.5-7B-Instruct` only and report results on both FaithEval and RealtimeQA. While the GLiNER backend currently only supports `Qwen2.5-7B-Instruct`, the pipeline can be evaluated on the other datasets as is.

## 2. Methodology

## 3. Main Results

## 4. How to Reproduce

### Requirements

- Python 3.11
- CUDA-capable GPU with enough VRAM for `Qwen/Qwen2.5-7B-Instruct` and `knowledgator/gliner-relex-large-v0.5`
- Git and Conda or a different Python environment manager

All experiments were run on a single NVIDIA A100 (Colab and BwUniCluster3.0) or H100 (BwUniCluster3.0) GPU, based on availability.

### Installation

```bash
git clone https://github.com/MarioKuzmanov/TruthfulRAG.git
cd TruthfulRAG

conda create -n truthfulrag-gliner python=3.11
conda activate truthfulrag-gliner

pip install -r requirements.txt
```

### Small experiment example

To test TruthfulRAG with GLiNER-Based KG Construction on 5 datapoints of the RealtimeQA dataset, run:

```bash
python -m gliner_truthfulrag.main \
  --experiment-id gliner-test-5 \
  --kg-backend gliner \
  --dataset-id timeqa_2022_nota.json \
  --dataset-limit 5
```

The results will be saved under:

```text
outputs/gliner-test-5/res.json
outputs/gliner-test-5/stats.jsonl
```

You can then generate a Markdown summary with:

```bash
python -m gliner_truthfulrag.eval \
  --experiment-id gliner-test-5
```

### Compare our method with the original

```bash
# GLiNER knowledge graph backend
python -m gliner_truthfulrag.main \
  --experiment-id timeqa-gliner \
  --kg-backend gliner \
  --dataset-id timeqa_2022_nota.json


# LLM knowledge graph backend
python -m gliner_truthfulrag.main \
  --experiment-id timeqa-qwen \
  --kg-backend llm \
  --dataset-id timeqa_2022_nota.json
```

In order to run the experiments on FaithEval change the dataset argument to:

```bash
--dataset-id faitheval_data.json
```

We also include the following options to run additional experiments:

```bash
# a KG construction pipeline vs. paths are directly generated by LLM based on context and query
--pipeline-mode {kg,direct-paths} 

# GLiNER backend vs. LLM backend
--kg-backend {gliner,llm}

# entropy filter used in original repo vs. the actual entropy filter from the original paper
--entropy-filter-method {legacy,paper}

# generate with paths + context vs. generate with paths only
--generation-context {original,none}

# manually set the entropy filter threshold (default is 3 for Qwen with legacy entropy filter)
--threshold FLOAT

# manually limit the size of the dataset
--dataset-limit INTEGER
```

### Run the demonstration API

TODO



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
