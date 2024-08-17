from fastapi import APIRouter, Depends
from typing import Annotated

from langchain.embeddings.base import Embeddings
from factory.embedding_factory import EmbeddingFactory
from services.file import FileHandler
from factory.file_handler_factory import S3Factory
from services.vector_store import VectorStore
from models.query_result import Metadata, QueryDocument, QueryResult

router = APIRouter(prefix="/query",)

file_handler_factory = S3Factory()
embedding_factory = EmbeddingFactory()

def get_file_handler() -> FileHandler:
    return file_handler_factory.get_service()

def get_embedding_model() -> Embeddings:
    return embedding_factory.get_service()

def transform_response(response):
    return [
            QueryResult(
                document=QueryDocument(
                    metadata=Metadata(
                        user=doc.metadata['user'],
                        date=doc.metadata['date']
                    ),
                    content=doc.page_content
                ),
                score=score
            )
            for doc, score in response
        ]

@router.get("/")
def query_chats(query: str, file_id: str, file_handler: Annotated[FileHandler, Depends(get_file_handler)], embedding_model: Annotated[Embeddings, Depends(get_embedding_model)]):
    store = VectorStore(file_id=file_id,
                        file_handler=file_handler,
                        embedding_model=embedding_model)
    response = store.similarity_search(query)
    print(f"Similar search: {response=}")
    return {"Similar queries": transform_response(response)}
