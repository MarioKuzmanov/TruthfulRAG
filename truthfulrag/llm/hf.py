# hf.py
import os
import copy
import asyncio
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Union
from util import logger
import numpy as np

import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError
)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

@lru_cache(maxsize=5)
def initialize_hf_client(
    model_name: str, 
    device_map: str = "auto", 
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> tuple:
    logger.info(f"Loading Hugging Face model: {model_name}")

    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config_kwargs = {
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
        }
        if load_in_4bit:
            quantization_config_kwargs.update(
                {
                    "bnb_4bit_compute_dtype": torch_dtype,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                }
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            **quantization_config_kwargs
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer, generation_config
    
    except (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError) as e:
        logger.error(f"Model loading error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise

def format_chat_messages(
    messages: List[Dict[str, str]], 
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None
) -> str:
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    for msg in messages:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        if "apply_chat_template" in dir(tokenizer):
            return tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    except Exception:
        logger.warning("Chat template application failed, using fallback formatting")
    
    return "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in formatted_messages]
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (RuntimeError, OSError, torch.cuda.OutOfMemoryError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def hf_chat_completion(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    **generation_params
) -> str:
    model, tokenizer, generation_config = initialize_hf_client(model_name)
    
    input_text = format_chat_messages(
        history_messages + [{"role": "user", "content": prompt}],
        tokenizer,
        system_prompt
    )
    
    params = generation_config.to_dict()
    params.update({
        "max_new_tokens": generation_params.get("max_new_tokens", 512),
        # "temperature": generation_params.get("temperature", 1e-6),
        # "top_p": generation_params.get("top_p", 0.9),
        "num_return_sequences": 1,
        "do_sample": generation_config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })
    
    generation_params.pop("hashing_kv", None)
    generation_params.pop("keyword_extraction", None)
    
    params.update(generation_params)
    
    def _sync_generate():
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=8192  
        ).to(model.device)
        
        stop_token_ids = [tokenizer.eos_token_id]
        if tokenizer.pad_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                **params
            )
        
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
    
    return await asyncio.to_thread(_sync_generate)

async def hf_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = [],
    model_name: Optional[str] = None,
    keyword_extraction: bool = False,
    **kwargs
) -> Union[str, Dict]:

    if not model_name:
        if "hashing_kv" in kwargs and "global_config" in kwargs["hashing_kv"]:
            model_name = kwargs["hashing_kv"]["global_config"]["llm_model_name"]
        else:
            raise ValueError("Model name not provided and could not be obtained from configuration")
    
    result = await hf_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
    
    return result

async def hf_embed(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()

async def hf_generate_with_logits(
    model_name: str,
    prompt: str,
    system_prompt: Optional[str],
    history_messages: List[Dict],
    top_k_tokens: int = 10,
    **generation_params
) -> List[List[Dict]]:
    model, tokenizer, generation_config = initialize_hf_client(model_name)

    input_text = format_chat_messages(
        history_messages + [{"role": "user", "content": prompt}],
        tokenizer,
        system_prompt
    )
    
    params = generation_config.to_dict()
    params.update({
        "max_new_tokens": generation_params.get("max_new_tokens", 512),
        "num_return_sequences": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
    })
    
    generation_params.pop("hashing_kv", None)
    generation_params.pop("keyword_extraction", None)

    params.update(generation_params)
    
    def _sync_generate():
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(model.device)

        stop_token_ids = [tokenizer.eos_token_id]
        if tokenizer.pad_token_id:
            stop_token_ids.append(tokenizer.pad_token_id)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                **params
            )
        return outputs.scores

    scores = await asyncio.to_thread(_sync_generate)

    topk_logit_lists = []

    for step_scores in scores[:generation_params.get("max_new_tokens", 1)]:
        values, indices = torch.topk(step_scores[0], top_k_tokens)
        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        top_log_probs = log_probs[indices]
        tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
        topk_logit_lists.append(
            list({"token": tok, "logprob": lp.item()} for tok, lp in zip(tokens, top_log_probs))
        )
    return topk_logit_lists