# TruthfulRAG with GLiNER-Based Knowledge Graph Construction

This project aims to extend [**TruthfulRAG**](https://github.com/STAIR-BUPT/TruthfulRAG/tree/main) by replacing the
LLM-based Knowledge Graph Construction with a faster pipeline, based on [**GLiNER**](https://github.com/urchade/GLiNER)
(Generalist and Lightweight Model for Named Entity Recognition).

## 1. Project Overview and Research Questions

Retrieval-Augmented Generation (RAG) provides Large Language Models (LLMs) with external source data before generation.
Nonetheless, those recovered sources may contain knowledge which conflicts with the knowledge the LLM learns during
training. The paper [TruthfulRAG](https://arxiv.org/abs/2511.10375) presents a framework to resolving these
factual-level conflicts. The author's approach entails the construction of Knowledge Graphs (KGs) from
subject-predicate-object triples extracted from the external source, two-hop graph traversal for reasoning path
retrieval, path relevancy ranking based on the query and entropy-based path filtering before final generation.

In the original implementation, large parts of the KG construction step (i.e., entity extraction, relation-label
generation, and triple extraction) are performed by an LLM. This requires multiple LLM inference calls, making the
pipeline resource- and time-intensive. Additionally, we identified several inconsistencies:

- **Entropy-filtering discrepancy:** The formulas described in Equations (9) and (11) in the paper differ from the
  formula implemented in the official repository.
- **Final-generation discrepancy:** Equation (12) in the paper does not mention the entire source context, although the
  implementation provides it alongside the reasoning paths during final generation.

Our main contribution is the replacement of the costly LLM-based KG backend with a faster backend based on GLiNER-relex,
a 500M-parameter specialized model for zero-shot Named Entity Recognition and Relationship Extraction. The remainder of
the TruthfulRAG pipeline, which includes graph retrieval, entropy-based filtering, final generation and evaluation,
remains unchanged, allowing a fair comparison between both KG construction methods.

This project primarily aims to address the following questions:

1. **Reproduction**: Can we replicate the results reported in the original paper?
2. **Efficiency and Quality**: How does our GLiNER-based pipeline compare to the TruthfulRAG pipeline, in regards to
   answer quality and runtime?
3. **Role of Reasoning Paths**: To what extent does the inclusion of reasoning paths affect answer quality, with and
   without appending the entire original context.

Additionally, we partly address further questions. These however are not the main focus of this project and can be
further investigated in the future:

4. **Entropy-filtering consistency**: How does the entropy-filtering method described in the paper compare with the
   method currently implemented in the official repository?
5. **Necessity of KG Construction**: Can query-aware reasoning paths be extracted directly from the source context while
   preserving answer quality and reducing runtime?
6. **Practical Deployability**: Can the proposed pipeline be integrated into TruthfulRAG as a deployable service?

> **Disclaimer**: The original paper reports experiments on the datasets FaithEval, MuSiQue, RealtimeQA and SQuAD across
> the three models `GPT-4o-mini`, `Qwen2.5-7B-Instruct` and `Mistral-7B-Instruct`. Due to time and resource constraints,
> we chose to run all experiments on `Qwen2.5-7B-Instruct` only and report results on both FaithEval and RealtimeQA.
> While
> the GLiNER backend currently only supports `Qwen2.5-7B-Instruct`, the pipeline can be evaluated on the other datasets
> as
> is.

## 2. Methodology

Our method consists of three main steps. The implementation is limited by the TruthfulRAG retrieval logic
The approach, together with the preliminary study are thoroughly described in [implementation.md](gliner_truthfulrag/implementation.md). The prompt for predicate extraction is [raw_predicate_extraction.md](gliner_truthfulrag/prompts/raw_predicate_extraction.md).

## 3. Main Results

- **Can we replicate the results reported in the original paper?**

The table reports the accuracy scores presented in the original paper and compares them with the results we got from
running the published code. All methods use `Qwen2.5-7B-Instruct` as LLM. Their baseline methods predict `wo_cot`, but
we also report their scores
with `cot`.

| Method      | TruthfulRAG-paper | TruthfulRAG-replicated | Dataset    | 
|-------------|-------------------|------------------------|------------|
| LLM         | 40.7%             | 43.4%                  | RealtimeQA |  
| LLM-cot     | -                 | 37.2%                  | RealtimeQA |
| RAG         | 78.7%             | 82.3%                  | RealtimeQA |
| RAG-cot     | -                 | 84.1%                  | RealtimeQA |
| TruthfulRAG | 82.3%             | 80.1%                  | RealtimeQA |
| TruthfulRAG | 73.2%             | 73.5%                  | FaithEval  |

Our replication yields similar results on FaithEval and somewhat close on RealtimeQA for TruthfulRAG. Surprisingly, RAG
performs better with and without CoT. The trend has to be validated on the other datasets.

- **How does our GLiNER pipeline compares to the TruthfulRAG's KG building?**

We continue the comparisons based on our replicated results. Also, the reported runtimes are measured within the same
GPU
environment to ensure fair comparisons. Filtered paths are the total number of filtered facts for the dataset.

| Method               | Accuracy | KG Building Runtime | Total Runtime | Filtered Paths | Dataset    | 
|----------------------|----------|---------------------|---------------|----------------|------------|
| TruthfulRAG-GLiNER   | 81.7%    | 540 s               | 1304 s        | 619            | RealtimeQA |
| TruthfulRAG-Original | 80.1%    | 4855 s              | 5612 s        | 534            | RealtimeQA |
| TruthfulRAG-GLiNER   | 76.6%    | 4452 s              | 12926 s       | 7678           | FaithEval  |
| TruthfulRAG-Original | 73.5%    | 43170 s             | 51678 s       | 7815           | FaithEval  |

So, GLiNER improves the accuracy by up to 3.1 percentage points, while being 9-9.7x faster at KG building.

- **What are the effects of reasoning paths?**

The baseline is provided with both contexts and paths. `wo-context` provides only the final paths without the source
context. The methods are only evaluated on RealtimeQA.

| Method                          | Accuracy | 
|---------------------------------|----------|
| TruthfulRAG-GLiNER-Baseline     | 81.7%    | 
| TruthfulRAG-Original-Baseline   | 80.1%    |
| TruthfulRAG-GLiNER-wo-context   | 53.9%    |
| TruthfulRAG-Original-wo-context | 57.5%    | 

The huge drop in both methods clearly indicates the reliance on context beyond facts.
Motivated by the huge important of context, we implement an approach that skips the KG building path to directly extract
and
provide triples for generating responses.

| Method                 | Accuracy | Path Extraction Runtime | Total Runtime | Avg. Filtered Paths | Dataset    | 
|------------------------|----------|-------------------------|---------------|---------------------|------------|
| RAG-triples            | 81.4%    | 164 s                   | 365 s         | 157                 | RealtimeQA |
| RAG-triples-wo-context | 57.5%    | 161 s                   | 319 s         | 157                 | RealtimeQA |
| RAG-triples            | 75.7%    | 2709 s                  | 5041 s        | 3302                | FaithEval  |

Without the context, the same drop is observed. However, RAG + triples has very slightly worse performance than GLiNER,
while being 2-3x faster.

All tables present averages where multiple runs are performed. For runtimes, when the runs were performed in different
compute environments, we only average from one environment. The individual outcomes can be found at `outputs/`.

## 4. How to Reproduce

### Requirements

- Python 3.11
- CUDA-capable GPU with enough VRAM for `Qwen/Qwen2.5-7B-Instruct` and `knowledgator/gliner-relex-large-v0.5`
- Git and Conda or a different Python environment manager

All experiments were run on a single NVIDIA A100 (Colab and BwUniCluster3.0) or H100 (BwUniCluster3.0) GPU, based on
availability.

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

The demo service requires a few lightweight dependencies, it does not run any LLMs in real-time and does not require
GPU.

However,
the recommended way of doing so is to create a new fresh environment (assumes conda is available):

```
conda create -n truthful-rag-service python=3.11 
conda activate truthful-rag-service 
```

Then, install the requirements which are in the `app/` directory:

`pip install -r app/requirements.txt`

Finally, for convenience use the `Makefile` to start the FastAPI server (from the repository root):

`make start`

By default, the Swagger UI will be available at: `http://0.0.0.0:8000/docs#/`. We implement tests for each endpoint on
the service, which can be started with `python -m unittest -v app.tests.test_endpoints` while the service is running. If
the binded url changes, the `URL` in the test file should be changed, too.

The demo exposes the intermediate steps, including prompts, extracted paths, reasoning and final responses of all
evaluated methods on a small set of 5 examples. It requires to be started with a single worker because we are
sharing it through all methods to directly compare responses.

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

## AI-Assistance statement
