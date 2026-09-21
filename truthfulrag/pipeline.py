import asyncio
import os
import shutil
import json
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, List, Optional, Type, cast
from datasets import Dataset
from datetime import datetime
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModel, AutoTokenizer
from .evaluate import (
    exact_match_score,
    acc_score,
    f1_score,
    metric_max_over_ground_truths
)
from .modules import (
    EntropyFilterMethod,
    chunking_by_token_size,
    generate_knowledge_graph,
    retrieve_knowledge_graph,
    entropy_filter,
    predict_answer,
    predict_answer_wo_facts
)
from .util import (
    FormatConverter
)
from .utils import (
    split_by_sentence,
    limit_async_func_call,
    compute_mdhash_id,
    json_converter,
    EmbeddingFunc
)
from truthfulrag.llm.hf import (
    hf_embed,
)

from .base import (
    StorageNameSpace,
)

STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
}

def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


@dataclass
class TruthfulRAG:
    """Truthful Retrieval-Augmented Generation pipeline"""
    dataset: Dataset
    backend_type: str
    model_name: str
    similarity_model: str
    threshold: float
    kg_making_sampling_params: Optional[Dict] = None
    kg_retrieval_sampling_params: Optional[Dict] = None
    entropy_filtering_sampling_params: Optional[Dict] = None
    generation_sampling_params: Optional[Dict] = None
    output_dir: str = field(
        default_factory=lambda: f"./truthfulrag_output_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    working_dir: str = field(
        default_factory=lambda: f"./truthfulrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    embedding_func: EmbeddingFunc = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    entropy_filter_method: EntropyFilterMethod = "legacy" # or "paper"

    kg_backend: str = "llm"
    gliner_kg_service_config: Dict = field(
        default_factory=lambda: {
            "gliner_model_id": "knowledgator/gliner-relex-large-v0.5",
            "llm_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "gliner_batch_size": 32,
            "llm_batch_size": 4
        }
    )

    def __post_init__(self):
        if self.kg_backend not in {"llm", "gliner"}:
            raise ValueError("kg_backend must be 'llm' or 'gliner'")
        if self.entropy_filter_method not in {"legacy", "paper"}:
            raise ValueError("entropy_filter_method must be 'legacy' or 'paper'")

        # Runtime objects stay out of dataclass fields: asdict() deep-copies them.
        # Load the service lazily, once, and reuse it across dataset items.

        self._gliner_kg_service = None
        if self.kg_backend == "gliner":
            self._gliner_kg_service = self._get_gliner_kg_service()

        # Set default sampling parameters if not provided
        if self.kg_making_sampling_params is None:
            self.kg_making_sampling_params = (
                {'max_tokens': 1000, 'top_p': 1.0} if self.backend_type == 'openai'
                else {'max_new_tokens': 1000, 'do_sample': False}
            )
        if self.kg_retrieval_sampling_params is None:
            self.kg_retrieval_sampling_params = (
                {'max_tokens': 1000, 'top_p': 1.0} if self.backend_type == 'openai'
                else {'max_new_tokens': 1000, 'do_sample': False}
            )
        if self.entropy_filtering_sampling_params is None:
            self.entropy_filtering_sampling_params = (
                {'top_p': 1.0} if self.backend_type == 'openai'
                else {'do_sample': False}
            )
        if self.generation_sampling_params is None:
            self.generation_sampling_params = (
                {'max_tokens': 1000, 'top_p': 1.0} if self.backend_type == 'openai'
                else {'max_new_tokens': 1000, 'do_sample': False}
            )

        time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        result_id = compute_mdhash_id(self.model_name + time)
        self.working_dir = os.path.join(self.working_dir, result_id)

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        global_config = asdict(self)
        self.vector_db_storage_cls = self._get_storage_class(self.vector_storage)
        self.graph_storage_cls = self._get_storage_class(self.graph_storage)

        self.vector_db_storage_cls = partial(self.vector_db_storage_cls, global_config=global_config)
        self.graph_storage_cls = partial(self.graph_storage_cls, global_config=global_config)

        if not self.embedding_func:
            self.embedding_func = EmbeddingFunc(
                embedding_dim=384,
                max_token_size=1000,
                func=lambda texts: hf_embed(
                    texts,
                    tokenizer=AutoTokenizer.from_pretrained(self.similarity_model),
                    embed_model=AutoModel.from_pretrained(self.similarity_model),
                ),
            )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.entity_name_vdb = self.vector_db_storage_cls(
            namespace="entities_name",
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=global_config,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )


    def _get_storage_class(self, storage_name: str) -> dict:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class


    async def reset_storages(self):
        await self.chunk_entity_relation_graph.clear_graph()

        await self.entities_vdb.clear_all_vectors()
        await self.entity_name_vdb.clear_all_vectors()
        await self.relationships_vdb.clear_all_vectors()


    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    # GLiNER-replacement service
    def _get_gliner_kg_service(self):
        if self._gliner_kg_service is None:
            from gliner_truthfulrag.services.kg_service import KGService
            self._gliner_kg_service = KGService(**self.gliner_kg_service_config)

        return self._gliner_kg_service

    async def make_knowledge_graph(
        self,
        sample: Dict,
        **generation_params
    ):
        """
        Generate knowledge graph for the dataset
        
        Args:
            dataset: Input dataset
            generation_params: Override parameters for generation

        Returns:
            List of knowledge graph dictionaries
        """
        if self.kg_backend == "gliner" and generation_params:
            raise ValueError("Generation parameters apply only to kg_backend='llm'")

        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)

        await self.reset_storages()

        # integrate GLiNER backend
        if self.kg_backend == "gliner":
            from gliner_truthfulrag.adapter import make_gliner_kg

            kg_service = self._gliner_kg_service
            await make_gliner_kg(
                sample,
                kg_service=kg_service,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entities_vdb=self.entities_vdb,
                entity_name_vdb=self.entity_name_vdb,
                relationships_vdb=self.relationships_vdb,
            )
        else:
            filtered_chunks = chunking_by_token_size(sample)
            params = {**self.kg_making_sampling_params, **generation_params}
            await generate_knowledge_graph(
                filtered_chunks,
                similarity_model=self.similarity_model,
                backend_type=self.backend_type,
                model_name=self.model_name,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entities_vdb=self.entities_vdb,
                entity_name_vdb=self.entity_name_vdb,
                relationships_vdb=self.relationships_vdb,
                **params
            )
        return await self._insert_done()


    async def knowledge_graph_retrieve(
        self,
        sample: Dict,
        **generation_params
    ):
        """
        Retrieve top-k triples from the knowledge graph for each query in the dataset
        
        Args:
            dataset: Input dataset
            generation_params: Override parameters for generation

        Returns:
            List of dictionaries with top-k elements for each query
        """
        params = {**self.kg_retrieval_sampling_params, **generation_params}
        # Get top-k triples
        elements = await retrieve_knowledge_graph(
            sample,
            similarity_model=self.similarity_model,
            backend_type=self.backend_type,
            model_name=self.model_name,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            entity_name_vdb=self.entity_name_vdb,
            relationships_vdb=self.relationships_vdb,
            **params
        )
        return elements


    async def entropy_based_filter(
        self,
        sample: Dict,
        elements: List[Dict],
        top_k: int = 10,
        threshold: float = 1,
        **generation_params
    ) -> List[Dict]:
        """
        Filter elements based on entropy threshold
        
        Args:
            elements: List of elements to filter
            threshold: Entropy threshold for filtering
            generation_params: Override parameters for generation
            
        Returns:
            Filtered list of elements
        """
        # Use provided parameters or defaults
        params = {**self.entropy_filtering_sampling_params, **generation_params}

        # Apply entropy filter
        return await entropy_filter(
            sample,
            elements,
            backend_type=self.backend_type,
            model_name=self.model_name,
            top_k=top_k,
            threshold=threshold,
            entropy_filter_method=self.entropy_filter_method,
            **params
        )


    async def get_predictions(
        self,
        dataset: Dataset,
        elements: List[Dict],
        generation_type: str = "cot",
        **generation_params
    ) -> Dict[str, str]:
        """
        Generate predictions for the dataset
        
        Args:
            dataset: Input dataset
            elements: List of elements to use for generation
            generation_type: Type of generation ("cot", "wo_cot")
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Use provided parameters or defaults
        params = {**self.generation_sampling_params, **generation_params}
        return await predict_answer(
            dataset,
            elements,
            backend_type=self.backend_type,
            model_name=self.model_name,
            generation_type=generation_type,
            **params
        )


    async def get_predictions_wo_elements(
        self,
        dataset: Dataset,
        withrag: bool = False,
        generation_type: str = "cot",
        **generation_params
    ) -> Dict[str, str]:
        """
        Generate predictions for the dataset
        
        Args:
            dataset: Input dataset
            withrag: Whether to use RAG for generation
            generation_type: Type of generation ("cot", "wo_cot")
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Use provided parameters or defaults
        params = {**self.generation_sampling_params, **generation_params}
        return await predict_answer_wo_facts(
            dataset,
            backend_type=self.backend_type,
            model_name=self.model_name,
            generation_type=generation_type,
            withrag=withrag,
            **params
        )


    async def run(
        self,
        dataset: Dataset,
        dataset_name: str
    ) -> Dict[str, str]:
        """
        Run the pipeline for the dataset
        
        Args:
            dataset: Input dataset
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        result_id = compute_mdhash_id(dataset_name + self.model_name)
        result_dir = os.path.join(self.output_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "result_name.txt"), "w") as f:
            f.write(f"dataset_name: {dataset_name}\n")
            f.write(f"model_name: {self.model_name}\n")

        elements_list = []
        filtered_elements_list = []
        for item in tqdm_asyncio(dataset, desc="Processing dataset items"):
            # Make knowledge graph
            await self.make_knowledge_graph(item)

            # Retrieve top-k triples
            elements = await self.knowledge_graph_retrieve(item)
            elements_list.append({"id": item['id'], "element": elements})
            print(f"elements: {elements}\n\n")
            # Filter elements based on entropy threshold
            filtered_elements = await self.entropy_based_filter(sample=item, elements=elements[:30], threshold=self.threshold)
            filtered_elements_list.append({"id": item['id'], "element": filtered_elements[:10]})
            print(f"filtered_elements: {filtered_elements}\n\n")
        with open(os.path.join(result_dir, "elements_list.json"), "w") as f:
            json.dump(elements_list, f, indent=4, default=json_converter, ensure_ascii=False)

        with open(os.path.join(result_dir, "filtered_elements_list.json"), "w") as f:
            json.dump(filtered_elements_list, f, indent=4, default=json_converter, ensure_ascii=False)

        # with open(os.path.join(result_dir, "filtered_elements_list.json"), "r") as f:
        #     filtered_elements_list = json.load(f)

        # Generate predictions
        predictions = await self.get_predictions(dataset, filtered_elements_list, generation_type="cot")
        with open(os.path.join(result_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=4, default=json_converter, ensure_ascii=False)
        results = self.evaluate(dataset, predictions, cot_format=True)
        with open(os.path.join(result_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4, default=json_converter, ensure_ascii=False)

        return results


    async def run_baseline(
        self,
        dataset: Dataset,
        dataset_name: str
    ) -> Dict[str, str]:
        """
        Run the pipeline for the dataset
        
        Args:
            dataset: Input dataset
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        result_id = compute_mdhash_id(dataset_name + self.model_name)
        result_dir = os.path.join(self.output_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "result_name.txt"), "w") as f:
            f.write(f"dataset_name: {dataset_name}\n")
            f.write(f"model_name: {self.model_name}\n")

        # Generate predictions
        predictions = await self.get_predictions_wo_elements(dataset, withrag=False, generation_type="wo_cot")
        with open(os.path.join(result_dir, "predictions_wo_rag.json"), "w") as f:
            json.dump(predictions, f, indent=4, default=json_converter, ensure_ascii=False)
        results = self.evaluate(dataset, predictions)
        with open(os.path.join(result_dir, "results_wo_rag.json"), "w") as f:
            json.dump(results, f, indent=4, default=json_converter, ensure_ascii=False)

        predictions = await self.get_predictions_wo_elements(dataset, withrag=True, generation_type="wo_cot")
        with open(os.path.join(result_dir, "predictions_w_rag.json"), "w") as f:
            json.dump(predictions, f, indent=4, default=json_converter, ensure_ascii=False)
        results = self.evaluate(dataset, predictions)
        with open(os.path.join(result_dir, "results_w_rag.json"), "w") as f:
            json.dump(results, f, indent=4, default=json_converter, ensure_ascii=False)

        return results


    async def run_ablation(
        self,
        dataset: Dataset,
        dataset_name: str
    ) -> Dict[str, str]:
        """
        Run the pipeline for the dataset
        
        Args:
            dataset: Input dataset
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        result_id = compute_mdhash_id(dataset_name + self.model_name)
        result_dir = os.path.join(self.output_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)
        ablation_dir = os.path.join(result_dir, "ablations")
        os.makedirs(ablation_dir, exist_ok=True)

        with open(os.path.join(ablation_dir, "result_name.txt"), "w") as f:
            f.write(f"dataset_name: {dataset_name}\n")
            f.write(f"model_name: {self.model_name}\n")

        with open(os.path.join(result_dir, "elements_list.json"), "r") as f:
            elements_list = json.load(f)

        # Generate predictions
        predictions = await self.get_predictions(dataset, elements_list, generation_type="cot")
        with open(os.path.join(ablation_dir, "predictions_wo_filtering.json"), "w") as f:
            json.dump(predictions, f, indent=4, default=json_converter, ensure_ascii=False)
        results = self.evaluate(dataset, predictions, cot_format=True)
        with open(os.path.join(ablation_dir, "results_wo_filtering.json"), "w") as f:
            json.dump(results, f, indent=4, default=json_converter, ensure_ascii=False)

        elements_list = []
        filtered_elements_list = []

        for item in tqdm_asyncio(dataset, desc="Processing dataset items"):
            # Filter elements based on entropy threshold
            elements = split_by_sentence(item['context'])
            elements_list.append({"id": item['id'], "element": elements})
            filtered_elements = await self.entropy_based_filter(sample=item, elements=elements[:30], threshold=self.threshold)
            filtered_elements_list.append({"id": item['id'], "element": filtered_elements[:10]})
            print(f"filtered_elements: {filtered_elements}\n\n")

        with open(os.path.join(ablation_dir, "filtered_elements_wo_kg_list.json"), "w") as f:
            json.dump(filtered_elements_list, f, indent=4, default=json_converter, ensure_ascii=False)

        # Generate predictions
        predictions = await self.get_predictions(dataset, filtered_elements_list, generation_type="cot")
        with open(os.path.join(ablation_dir, "predictions_wo_kg.json"), "w") as f:
            json.dump(predictions, f, indent=4, default=json_converter, ensure_ascii=False)
        results = self.evaluate(dataset, predictions, cot_format=True)
        with open(os.path.join(ablation_dir, "results_wo_kg.json"), "w") as f:
            json.dump(results, f, indent=4, default=json_converter, ensure_ascii=False)

        return results


    def evaluate(
        self,
        dataset: Dataset,
        predictions: Dict[str, str],
        cot_format: bool = False,
        detailed_output: bool = False
    ) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            dataset: Input dataset with ground truth
            predictions: Generated predictions
            detailed_output: Whether to include per-item details
            
        Returns:
            Evaluation results dictionary
        """
        prediction_details = []
        total_em = total_acc = total_f1 = 0
        num_items = 0

        for item in tqdm(dataset, desc="Evaluating"):
            prediction = predictions.get(item['id'], "")
            # if prediction is in JSON format, extract the 'answer' field
            if cot_format:
                prediction = FormatConverter.extract_answer(prediction)
            ground_truth = item['answer']
            if isinstance(prediction, list):
                prediction = prediction[0]

            print(f"prediction: {prediction}\n\n")

            # Calculate metrics
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
            acc_score_val = metric_max_over_ground_truths(
                acc_score, prediction, ground_truth)
            f1_score_val = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)

            # Accumulate totals
            total_em += em_score
            total_acc += acc_score_val
            total_f1 += f1_score_val
            num_items += 1

            # Store details if requested
            if detailed_output:
                prediction_details.append({
                    "id": item['id'],
                    "question": item['question'],
                    "answer": ground_truth,
                    "prediction": prediction,
                    "exact_match": em_score,
                    "acc": acc_score_val,
                    "f1": f1_score_val
                })

        # Calculate averages
        avg_em = 100.0 * total_em / num_items if num_items > 0 else 0
        avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
        avg_f1 = 100.0 * total_f1 / num_items if num_items > 0 else 0

        # Prepare result
        result = {
            "num_items": num_items,
            "exact_match": avg_em,
            "acc": avg_acc,
            "f1": avg_f1
        }

        if detailed_output:
            result["details"] = prediction_details

        return result