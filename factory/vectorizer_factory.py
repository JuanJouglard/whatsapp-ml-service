from abc import ABC, abstractmethod
from functools import lru_cache
from services.vectorize import VectorizerInterface, Vectorizer


class VectorizerAbstractFactory(ABC):

    @abstractmethod
    def get_service(self) -> VectorizerInterface:
        pass

class VectorizerFactory(VectorizerAbstractFactory):

    def __init__(self):
        pass

    @lru_cache()
    def get_service(self) -> Vectorizer:
        return Vectorizer()
