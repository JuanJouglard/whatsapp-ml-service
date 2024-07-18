from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


class VectorDatabase:
    def __init__(self, documents):
        self.documents = documents
        self.RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=doc["content"], metadata=doc["metadata"]) for doc in documents
        ]

    def create_store(self, docs_processed: LanchainDocument[], embedding_model: HuggingFaceEmbeddings, distance: DistanceStrategy = DistanceStrategy.COSINE):
        self.vector_store = FAISS(docs_processed, embedding_model, distance)

    def similarity_search(query: String, top_n: int):
        return self.vector_store.similarity_search(query)


