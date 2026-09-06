import asyncio
import os
from datasets import load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from truthfulrag import TruthfulRAG

from dotenv import dotenv_values

cfg = dotenv_values("gliner_truthfulrag/.env")


os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]
# os.environ["OPENAI_BASE_URL"] = ""


async def main():
    # Load dataset
    dataset_name = 'timeqa_2022_nota' # faitheval_data, musique_negative, squad_negative, timeqa_2022_nota, musique_golden, squad_golden
    dataset = load_dataset("json", data_files=f"./datas/{dataset_name}.json")
    dataset = dataset['train']
    dataset = dataset.select(range(0,1,1))

    rag = TruthfulRAG(
        dataset=dataset,
        backend_type="openai",  # or "hf", "ollama", "openai"
        model_name="gpt-4o-mini", # Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, gpt-4o-mini-ca
        similarity_model="sentence-transformers/all-MiniLM-L6-v2",
        output_dir="./results", # ./results ./results/baselines
        working_dir="./kg_cache",
        threshold=1
    )

    await rag.make_knowledge_graph(sample=dataset[0])
 
    # evaluation = await rag.run(dataset, dataset_name)
    # evaluation = await rag.run_baseline(dataset, dataset_name)
    # evaluation = await rag.run_ablation(dataset, dataset_name)

    #
    # print("Evaluation Results:")
    # print(f"Exact Match: {evaluation['exact_match']:.2f}%")
    # print(f"Accuracy: {evaluation['acc']:.2f}%")
    # print(f"F1 Score: {evaluation['f1']:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())