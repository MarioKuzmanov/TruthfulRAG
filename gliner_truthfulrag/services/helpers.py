from typing import Any
from pathlib import Path

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
from gliner import GLiNER
import torch
from huggingface_hub import snapshot_download

from gliner_truthfulrag.core.env_manager import settings

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)


def download_load_gliner(model_id: str) -> tuple[Any, PreTrainedTokenizerBase]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"GLiNER uses GPU, the set GPU for this experiment is {settings.DEVICE}")

    model = GLiNER.from_pretrained(model_id, map_location=settings.DEVICE).eval()

    tokenizer = model.data_processor.transformer_tokenizer
    encoder = model.model.token_rep_layer.bert_layer.model

    logger.info(
        "GLiNER: %s, Encoder: %s, Tokenizer: %s",
        type(model).__name__,
        type(encoder).__name__,
        type(tokenizer).__name__,
    )

    return model, tokenizer


def download_load_qwen(model_id: str) -> tuple[Any, Any]:
    if Path(model_id).is_dir():
        model_id_or_path = str(Path(model_id).absolute())
    else:
        # download model and tokenizer weights
        model_id_or_path = snapshot_download(model_id)

    # load models
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=settings.DEVICE, torch_dtype=torch.float16,
                                                 trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, device_map=settings.DEVICE, padding_side="left",
                                              trust_remote_code=True)
    return model, tokenizer


def init_pipeline(gliner_model_id: str, llm_model_id: str) -> tuple:
    gliner_model, gliner_tokenizer = download_load_gliner(model_id=gliner_model_id)
    qwen_model, qwen_tokenizer = download_load_qwen(model_id=llm_model_id)

    return gliner_model, gliner_tokenizer, qwen_model, qwen_tokenizer


def unload_model(model):
    logger.info("Before")
    free, total = torch.cuda.mem_get_info()

    logger.info(f"Total GBs: {total / 2 ** 30:.2f}")
    logger.info(f"Free GBs: {free / 2 ** 30:.2f}")

    del model

    torch.cuda.empty_cache()

    logger.info("After")

    free, total = torch.cuda.mem_get_info()

    logger.info(f"Total GBs: {total / 2 ** 30:.2f}")
    logger.info(f"Free GBs: {free / 2 ** 30:.2f}")
