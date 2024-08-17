from abc import ABC, abstractmethod
from functools import lru_cache
from langchain.embeddings.base import Embeddings
from torch.cuda import is_available
from services.embedding import RandomEmbeddingModel,HuggingFaceEmbeddings


class EmbeddingAbstractFactory(ABC):

    @abstractmethod
    def get_service(self) -> Embeddings:
        pass

class EmbeddingFactory(EmbeddingAbstractFactory):

    def __init__(self):
        pass

    def get_service(self) -> Embeddings:
        if is_available():
            return HuggingFaceEmbeddings()
        else:
            print("USING RANDOM EMBEDDING")
            return RandomEmbeddingModel()

