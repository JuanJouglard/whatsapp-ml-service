from abc import ABC, abstractmethod
import os

import pandas as pd
from transformers import AutoTokenizer

from services.file import S3Handler

from .embedding import EmbeddingHuggingFace, RandomEmbeddingModel
from .knowledge_base import LangchainKnowledgeBase
from .vector_store import VectorStore
from factory.embedding_factory import EmbeddingFactory
from logger import logger

class VectorizerInterface(ABC):

    @abstractmethod
    def setup_vectorizer(self, chats: pd.DataFrame):
        pass



class Vectorizer(VectorizerInterface):

    def setup_vectorizer(self, chats: pd.DataFrame):
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

        print(f"Setup vectorizer... {EMBEDDING_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

        print(f"Setup knowledge base...")
        knowledge_base = LangchainKnowledgeBase(chats)
        docs_splitted = knowledge_base.split_documents(128, tokenizer)
        print(f"split documents")

        print(f"Create embedding model")
        embedding_model = EmbeddingFactory().get_service()

        print(f"Create vector store")
        self.store = VectorStore(knowledge_base=docs_splitted,
                                 embedding_model=embedding_model,
                                 file_handler=S3Handler())

    def save_vector_store(self, chat_id: str):
        return self.store.save(chat_id)
