from datetime import datetime
from langchain.docstore.document import Document
from pydantic import BaseModel

class Metadata(BaseModel):
    user: str
    date: datetime

class QueryDocument(BaseModel):
    content: str
    metadata: Metadata

class QueryResult(BaseModel):
    document: QueryDocument
    score: float



