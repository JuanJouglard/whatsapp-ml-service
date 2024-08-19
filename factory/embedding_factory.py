from abc import ABC, abstractmethod
from functools import lru_cache
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available
from services.embedding import RandomEmbeddingModel, EmbeddingHuggingFace


class EmbeddingAbstractFactory(ABC):

    @abstractmethod
    def get_service(self) -> Embeddings:
        pass

class EmbeddingFactory(EmbeddingAbstractFactory):

    def __init__(self):
        pass

    def get_service(self) -> Embeddings:
        if is_available():
            print("[EMBEDDING] USING HUGGING_FACE EMBEDDING")
            return EmbeddingHuggingFace()
        else:
            print("[EMBEDDING] USING RANDOM EMBEDDING")
            return RandomEmbeddingModel()

