import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import json_repair
import torch

from gliner_truthfulrag.core.env_manager import settings

logger = logging.getLogger(__name__)


def parse_decoded(response: str) -> list[str]:
    try:
        paths = json.loads(response)
    except json.JSONDecodeError as error:
        logger.warning("Malformed direct-path JSON; attempting repair: %s", error)
        paths = json_repair.loads(response)

    if not isinstance(paths, list):
        raise ValueError("Direct-path response must be a JSON array")

    parsed_paths = []
    seen_paths = set()
    for index, path in enumerate(paths):
        if not isinstance(path, list):      # ensure each direct path is a list
            logger.warning("Direct path at index %d must be a JSON array", index)
            continue
        if len(path) not in {3, 5}:         # each direct path must contain exactly 3 or 5 elements
            logger.warning(
                "Direct path at index %d must contain exactly 3 or 5 strings",
                index,
            )
            continue
        if not all(isinstance(part, str) and part.strip() for part in path):    # each element in the direct path must be a non-empty string
            logger.warning(
                "Direct path at index %d must contain only non-empty strings",
                index,
            )
            continue

        normalized_path = tuple(part.strip() for part in path)
        if normalized_path in seen_paths:   # no duplicate direct paths allowed
            continue

        seen_paths.add(normalized_path)
        parsed_paths.append(" -> ".join(normalized_path))   # serialize and add arrow for direction
        if len(parsed_paths) == 30:     # enforce a maximum of 30 paths
            break

    return parsed_paths


# pre-load extraction prompt
with open(f"{settings.PROMPT_DIR}/direct_path_extraction.md", "r") as f:
    PROMPT = f.read()


class AbstractDirectPathService(ABC):
    @abstractmethod
    def extract_direct_paths(self, item: dict) -> list[str]:
        ...


class QwenDirectPathService(AbstractDirectPathService):
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def _format_chat_messages(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Question:\n{question}\nSource text:\n{context}"},
        ]
        messages_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return messages_prompt

    def extract_direct_paths(self, item: dict) -> list[str]:
        question = item.get("question")
        context = item.get("context")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("item must contain a non-empty question")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("item must contain non-empty context")

        prompt = self._format_chat_messages(question, context)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Deterministic
        sampling_params = {
            "max_new_tokens": 1000,
            "do_sample": False,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1
        }

        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)
            outputs = self.model.generate(**inputs, **sampling_params)
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if len(decoded) != 1:
            raise RuntimeError(f"Expected one direct-path response, received {len(decoded)}")

        return parse_decoded(decoded[0])