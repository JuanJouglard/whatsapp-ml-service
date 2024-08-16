from abc import ABC, abstractmethod
import boto3
import os
from io import BytesIO
import pandas as pd
import os

class FileHandler(ABC):

    @abstractmethod
    def read_folder(self, bucket: str, file_id: str):
        pass

    @abstractmethod
    def read_file(self, file_id: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_file(self, content: str) -> str:
        pass

    @abstractmethod
    def save_folder(self, folder_path: str):
        pass


class S3Handler(FileHandler):

    def __init__(self):
        ENDPOINT = os.getenv("ENDPOINT")
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        self.client = boto3.client('s3',
                                 endpoint_url=ENDPOINT,
                                 aws_access_key_id=AWS_ACCESS_KEY,
                                 aws_secret_access_key=AWS_SECRET_KEY,
                                 aws_session_token=None,
                                 config=boto3.session.Config(signature_version='s3v4'),
                                 verify=False)

    def read_file(self, file_id: str) -> pd.DataFrame:
        BUCKET_NAME = "conversations"

        file_stream = BytesIO()

        self.client.download_fileobj(BUCKET_NAME, f"{file_id}.pkl", file_stream)
        file_stream.seek(0)

        return pd.DataFrame(pd.read_pickle(file_stream))

    def read_folder(self, bucket: str, file_id: str):
        pass

    def save_folder(self, folder_path: str):
        BUCKET_NAME = "vectorstores"

        for root, _, files in os.walk(folder_path):
            for file in files:
                print(f"File: {file}")
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, folder_path)
                s3_path = os.path.join(folder_path, relative_path)

                # Upload the file to S3
                self.client.upload_file(local_path, BUCKET_NAME, s3_path)


    def save_file(self, content: str) -> str:
        return content


