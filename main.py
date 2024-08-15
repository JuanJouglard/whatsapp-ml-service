from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from services.file import read_file_from_bucket
from services.vectorize import Vectorizer

load_dotenv()

app = FastAPI()

class ChatData(BaseModel):
    file_id: str

@app.get("/chats")
def query_chats(query: str):
    print(f"query: {query}")
    return {"Body": "Query chats"}

@app.post("/chats")
async def parse_chat(chat: ChatData):
    print(f"file id: {chat}")
    dataframe = await read_file_from_bucket(chat.file_id)
    vectorizer = Vectorizer(dataframe)
    vectorizer.save_vector_store(chat.file_id)
    return {"Body": "Parse chats"}
