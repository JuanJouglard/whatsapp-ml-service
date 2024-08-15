import boto3
import os
import time
from io import BytesIO
import pandas as pd

async def read_file_from_bucket(file_id: str):
    print(os.environ)
    ENDPOINT = os.getenv("ENDPOINT")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    BUCKET_NAME = "conversations"

    file_stream = BytesIO()

    s3_client = boto3.client('s3',
                             endpoint_url=ENDPOINT,
                             aws_access_key_id=AWS_ACCESS_KEY,
                             aws_secret_access_key=AWS_SECRET_KEY,
                             aws_session_token=None,
                             config=boto3.session.Config(signature_version='s3v4'),
                             verify=False)

    s3_client.download_fileobj(BUCKET_NAME, f"{file_id}.pkl", file_stream)
    file_stream.seek(0)

    return pd.read_pickle(file_stream)
