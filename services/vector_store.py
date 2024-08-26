from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Annotated

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings

from services.file import FileHandler


class VectorStoreInterface(ABC):

    @abstractmethod
    def similarity_search(self, query: str) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def save(self, file_id: str) -> Annotated[str, "File name"]:
        pass


class VectorStore(VectorStoreInterface):

    def __init__(self, embedding_model: Embeddings, file_handler: FileHandler, knowledge_base: Optional[List[Document]] = [], file_id: str = ""):
        self.file_handler = file_handler
        if file_id:
            print("Loading vector store")
            folder_path = self.file_handler.read_folder("vectorstores", f"{file_id}.faiss")
            self.store = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
        else:
            self.store = FAISS.from_documents(knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, query: str) -> List[Tuple[Document, float]]:
        print(f"Running similarity search with {query=}")
        return self.store.similarity_search_with_score(query)

    def save(self, file_id: str) -> Annotated[str, "File name"]:
        print("save store")
        path = f"{file_id}.faiss"
        self.store.save_local(path)
        self.file_handler.save_folder(path)
        return path
