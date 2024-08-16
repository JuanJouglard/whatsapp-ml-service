import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer


class Embedding:

    def __init__(self):
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
        self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
                )
        self._tokenizer = AutoTokenizer.from_pretrained_model(EMBEDDING_MODEL_NAME)

    @property
    def model(self):
        return self._embedding_model

    @property
    def tokenizer(self):
        return self._tokenizer

    def embed_query(query: String):
        return self._embedding_model.embed_query(query)



