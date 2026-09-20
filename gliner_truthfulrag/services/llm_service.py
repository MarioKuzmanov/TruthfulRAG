from abc import ABC, abstractmethod
import json
from typing import Any
import torch

from gliner_truthfulrag.core.env_manager import settings


def parse_decoded(response: str) -> list[str]:
    try:
        predicates = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError(f"Predicates from {response} not parsed")

    return list(dict.fromkeys(predicates))


# pre-load extraction prompt
with open(f"{settings.PROMPT_DIR}/raw_predicate_extraction.md", "r") as f:
    PROMPT = f.read()


class AbstractLLMService(ABC):
    @abstractmethod
    def extract_raw_predicates(self, chunks: list) -> list[str]:
        ...


class QwenLLMService(AbstractLLMService):
    def __init__(self, model: Any, tokenizer: Any, batch_size: int):
        self.model, self.tokenizer = model, tokenizer
        self.batch_size = batch_size

    def _format_chat_messages(self, text: str):
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Source text:\n\n{text}"},
        ]

        messages_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return messages_prompt

    def extract_raw_predicates(self, chunks: list) -> list[str]:
        # batch-inference on GPU
        # chunk = {item_id: str, tokens: int, chunk_id: str, context: str}

        texts = [chunk["context"] for chunk in chunks]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        sampling_params = {
            "max_new_tokens": 1000,
            "do_sample": False,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1
        }

        predicates = []

        with torch.no_grad():

            for i in range(0, len(texts), self.batch_size):
                texts_batch = texts[i: i + self.batch_size]

                prompts = [self._format_chat_messages(text) for text in texts_batch]
                inputs_batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False,
                                              add_special_tokens=False).to(self.model.device)

                outputs = self.model.generate(**inputs_batch, **sampling_params)

                generated_ids = outputs[:, inputs_batch["input_ids"].shape[1]:]

                decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for predicate_list in decoded:
                    predicates_parsed = parse_decoded(predicate_list)
                    predicates.extend(predicates_parsed)

        return predicates
