
from fastapi import FastAPI

app = FastAPI()



@app.get("/chats")
def query_chats(query: String):
    return {"Body": "Query chats"}
