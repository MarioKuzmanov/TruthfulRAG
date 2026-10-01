# Statement on AI usage

### Mario:

I have used only [ChatGPT](https://chatgpt.com/) by giving some context, end goal in mind and directly asking questions, or pasting snippets of code to be better understood, analyzed, refined, or improved.

Here are the main use-cases I found for it during the project implementation:

1. Diagram/Figure generation

Example:

"""
Here is the approach, please generate a mermaid diagram that describes these steps:

STEP: Chunker Service

token-based chunking with overlap
tokens are given from GLiNER's tokenizer
conceptually same as the one in TruthfulRAG
STEP: Raw Predicate Extraction

main LLM performs batched-raw predicate extraction from text chunks
STEP: NER and RE

GLiNER performs NER and RE based on the pre-defined entities and relations schemas
Batched inference with full-precision weights and optimized FlashDeBERTa kernel backend
STEP: Integration

adapter.py for integration into TruthfulRAG
native-support by customized kg_backend and gliner_kg_service_config
STEP: Evaluation

eval.py for report generation with summaries for the experimental runs
"""
And the produced (refined if necessary) mermaid diagram is used. 

This includes providing some conclusions about intermediate/final output results and etc. ChatGPT itself was not used to make decision what should be evaluated next or similar. 


2. Analysis and examples selection for the demo service

 - Here, I have given a subset of outputs of some experimental run for all methods and have asked ChatGPT to analyze them, then choose the 10 most illustrative one. Later, I have used them (or some of them) for the demo service. ChatGPT was not included for designing the architecture of the service, however it was asked for recommendations on how to best share an "item_id" across endpoints. 
 

3. Coding assistance

 - ChatGPT was mainly used for analyzing and generating snippets of code.
 
 - For the GLiNER integration to TruthfulRAG, the model was given selected methods and/or parts of scripts (like the GraphStorage implementation) and then asked to explain with code snippets what TruthfulRAG expects and what should GLiNER output. Here is an example:

here is the function _merge_nodes_then_upsert, also the nano_vector_db_impl.py and networkx_impl.py. How should my nodes and edges look like to be upserted?

And then the conversation might continue with more similar questions based on the response until I feel that I have understood it. 

However, no AI-generated code without refinements, even significant changes was directly used. ChatGPT did not decide the pipeline behind the GLiNER-based KG building, but it was familiar with the idea  and specific details of the steps implementation. However, its suggested changes were not considered.

### Alejandro:

I used an AI assistant through the Copilot SDK inside of Visual Studio Code during this project, to:

- understand and navigate the existing TruthfulRAG codebase
- understand the data flow throughout the pipeline
- suggest code snippets and debugging
- validate experiment scripts
- analyze experiment logs
- proof-read the documentation

All AI-generated recommendations, explanations and code were reviewed, adapted and, if needed, tested by me. It did not independetly determine research direction, methods, architecture, experiments and conclusions.


#### Example 1: 

**Input:**

What would I need to change in order to not give the context in addition to the paths in final generation?

**Output and use:**  
Copilot traced final generation through the pipeline functions. It mentioned where the original context and paths are passed until the final generation. It helped me implement the `--generation-context` by keeping an overview on cross-dependencies and argument passes inside the codebase.

#### Example 2: 

**Input:**

How long would the ablation with context need on FaithEval?

**Output and use:**  
Copilot estimated total runtime by analyzing filesize and prior experiment reports on RealtimeQA, suggesting times for the Slurm scripts I used to run jobs on the BwUniCluster3.0.

#### Example 3:

**Input:**

Maybe we can also just warn, but not break? Can you suggest a minimal change? (During the direct-path ablation a single malformed path would also discard all other correct paths for the current example.)

**Output and use:**  
Copilot suggested a minimal code change and using the json_repair package in the function, which I reviewed and consequentially tested on a single datapoint which was previously malformed and yielded no retained paths.