from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import boto3
import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY_ID = os.getenv("AWS_SECRET_KEY_ID")
ENDPOINT = os.getenv("S3_ENDPOINT")
BUCKET_NAME = "vectorstores"

class VectorDatabase:
    def __init__(self):
        self.s3_client = boto3.resource('s3',
                                      endpoint_url=ENDPOINT,
                                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                                      aws_secret_access_key=AWS_SECRET_KEY_ID,
                                      aws_session_token=None,
                                      config=boto3.session.Config(signature_version='s3v4'),
                                      verify=False
                         ).Bucket(BUCKET_NAME)

    def create_store(self, chat_id:String, docs_processed: LangchainDocument[], embedding_model: HuggingFaceEmbeddings, distance: DistanceStrategy = DistanceStrategy.COSINE):
        if self.vector_store := check_if_file_exists(chat_id):
            print("VECTOR STORE EXISTS")
        else:
            self.vector_store = FAISS(docs_processed, embedding_model, distance)

    def similarity_search(self, query: String, top_n: int):
        return self.vector_store.similarity_search(query)


    def check_if_file_exists(self, chat_id: String):
        return self.s3_client_bucket.download_file(Filename=f"{chat_id}.pkl", Key=f"{chat_id}.pkl")

    def save_file_to_bucket(self, chat_id):
        content = self.vector_store.serializer_to_bytes()
        return self.s3_client_bucket.put_object(Key=f"{chat_id}.pkl", Body=content)


