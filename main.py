from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated


from factory.file_handler_factory import S3Factory
from factory.vectorizer_factory import VectorizerFactory
from services.file import FileHandler
from services.vectorize import Vectorizer, VectorizerInterface
from routes import query

load_dotenv()

app = FastAPI()

origins = [
        "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)

file_handler_factory = S3Factory()
vectorizer_factory = VectorizerFactory()

class ChatData(BaseModel):
    file_id: str

def get_file_handler() -> FileHandler:
    return file_handler_factory.get_service()

def get_vectorizer() -> VectorizerInterface:
    return vectorizer_factory.get_service()

@app.post("/chats")
async def parse_chat(chat: ChatData,
                     file_handler: Annotated[FileHandler, Depends(get_file_handler)],
                     vectorizer: Annotated[Vectorizer, Depends(get_vectorizer)]):

    dataframe = file_handler.read_file(chat.file_id)
    vectorizer.setup_vectorizer(dataframe)
    vectorizer.save_vector_store(chat.file_id)
    return {"Body": "Parse chats"}
