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
    def chunk_by_token_size(self, item: dict) -> list[dict]:
        ...


MAX_TOKENS = 512
OFFSET = 30

OVERLAP_TOKEN_SIZE = 64
MAX_TOKEN_SIZE = MAX_TOKENS - OFFSET


class ChunkerService(AbstractChunkService):

    def __init__(self, tokenizer: DebertaV2Tokenizer):
        self.tokenizer = tokenizer

    def chunk_by_token_size(self, item: dict) -> list[dict]:
        context, item_id = item["context"], str(item["id"])

        tokens = self.tokenizer(context, add_special_tokens=False).input_ids

        chunks = []

        for index, start in enumerate(range(0, len(tokens), MAX_TOKEN_SIZE - OVERLAP_TOKEN_SIZE)):
            chunk_content = self.tokenizer.decode(tokens[start: start + MAX_TOKEN_SIZE], skip_special_tokens=True)

            chunks.append({
                "item_id": item_id,
                "tokens": min(MAX_TOKEN_SIZE, len(tokens) - start),
                "context": chunk_content.strip(),
                "chunk_id": f"{item_id}_{index}"
            })

        return chunks
