from abc import ABC, abstractmethod
import os

import pandas as pd
from transformers import AutoTokenizer

from .embedding import EmbeddingWrapper
from .knowledge_base import LangchainKnowledgeBase
from .vector_store import VectorStore
from logger import logger


class VectorizerInterface(ABC):

    @abstractmethod
    def setup_vectorizer(self, chats: pd.DataFrame):
        pass



class Vectorizer(VectorizerInterface):

    def __init__(self, chats: pd.DataFrame):
        self.setup_vectorizer(chats)

    def setup_vectorizer(self, chats: pd.DataFrame):
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
        print(f"Setup vectorizer... {EMBEDDING_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

        print(f"Setup knowledge base...")
        knowledge_base = LangchainKnowledgeBase(chats)
        docs_splitted = knowledge_base.split_documents(128, tokenizer)
        print(f"split documents: {docs_splitted}")

        print(f"Create embedding model")
        embedding_model = EmbeddingWrapper()

        print(f"Create vector store")
        self.store = VectorStore(docs_splitted, embedding_model.model)

    def save_vector_store(self, chat_id: str):
        self.store.save(chat_id)
