import os

from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class EmbeddingHuggingFace(Embeddings):

    def __init__(self):
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

        self._model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                encode_kwargs={"normalize_embeddings": True})

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self._model.embed_query(text)


class RandomEmbeddingModel(Embeddings):
    def __init__(self, dim=128):
        self.dim = dim

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return list(np.random.rand(self.dim))
