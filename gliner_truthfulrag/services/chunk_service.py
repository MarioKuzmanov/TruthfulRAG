from abc import ABC, abstractmethod
import logging

from transformers import DebertaV2Tokenizer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "[%(name)s] %(message)s"
    ),
)


class AbstractChunkService(ABC):
    @abstractmethod
    def chunk_by_gliner_token_size(self, item: dict) -> list[dict]:
        ...


class ChunkerService(AbstractChunkService):

    def __init__(self, tokenizer: DebertaV2Tokenizer, max_tokens: int, offset: int, overlap_tokens: int):
        self.tokenizer = tokenizer
        self.max_token_size = max_tokens - offset
        self.overlap_token_size = overlap_tokens

    def chunk_by_gliner_token_size(self, item: dict) -> list[dict]:
        logger.info(f"item={item}")
        context, item_id = item["context"], str(item["id"])

        tokens = self.tokenizer(context, add_special_tokens=False).input_ids

        chunks = []

        for index, start in enumerate(range(0, len(tokens), self.max_token_size - self.overlap_token_size)):
            chunk_content = self.tokenizer.decode(tokens[start: start + self.max_token_size], skip_special_tokens=True)

            chunks.append({
                "item_id": item_id,
                "tokens": min(self.max_token_size, len(tokens) - start),
                "context": chunk_content.strip(),
                "chunk_id": f"{item_id}_{index}"
            })

        return chunks
