from abc import ABC, abstractmethod


class AbstractRegistryService(ABC):
    @abstractmethod
    def register(self, item_id: str):
        ...


class LLMRegistryService(AbstractRegistryService):
    def register(self, item_id: str):
        ...


def get_registry_service(method_id: str) -> AbstractRegistryService:
    if method_id == "llm":
        return LLMRegistryService()
    else:
        raise NotImplementedError(f"Method {method_id} not implemented")
