from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from services.file import S3Handler
from services.vector_store import VectorStore
from services.vectorize import Vectorizer
from services.embedding import RandomEmbeddingModel

load_dotenv()

app = FastAPI()

class ChatData(BaseModel):
    file_id: str

@app.get("/chats")
def query_chats(query: str, file_id: str):
    print(f"query: {query}")
    store = VectorStore(file_id=file_id,
                        file_handler=S3Handler(),
                        embedding_model=RandomEmbeddingModel())
    response = store.similarity_search(query)
    print(f"Similar search: {response=}")
    return {"Similar queries": response}

@app.post("/chats")
async def parse_chat(chat: ChatData):
    file_handler = S3Handler()
    print(f"file id: {chat}")
    dataframe = file_handler.read_file(chat.file_id)
    vectorizer = Vectorizer(dataframe)
    vectorizer.save_vector_store(chat.file_id)
    return {"Body": "Parse chats"}
