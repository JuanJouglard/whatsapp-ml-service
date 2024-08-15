from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


class VectorStoreInterface(ABC):

    @abstractmethod
    def similarity_search(self, query: str):
        pass

    @abstractmethod
    def save(self, file_id: str):
        pass



class VectorStore(VectorStoreInterface):

    def __init__(self, knowledge_base: List[Document], embedding_model):
        self.store = FAISS.from_documents(knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, query: str):
        return self.store.similarity_search_with_score(query)

    def save(self, file_id: str):
        print("save store")
        self.store.save_local(f"{file_id}.faiss")

