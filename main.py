from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class ChatData(BaseModel):
    file_id: str

@app.middleware("http")
async def log_stuff(request: Request, call_next):
    print(f"{request.method} {request.url}")
    print(f"{await request.body()}")
    response = await call_next(request)
    print(response)
    print(response.status_code)
    return response

@app.get("/chats")
def query_chats(query: str):
    print(f"query: {query}")
    return {"Body": "Query chats"}

@app.post("/chats")
def parse_chat(chat: ChatData):
    print(f"file id: {chat}")
    return {"Body": "Parse chats"}
