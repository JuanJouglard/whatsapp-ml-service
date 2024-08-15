import os
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingWrapper:

    def __init__(self):
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

        self._model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                encode_kwargs={"normalize_embeddings": True})

    @property
    def model(self):
        return self._model
