from abc import ABC, abstractmethod
from functools import lru_cache
from services.file import S3Handler, FileHandler


class FileHandlerAbstractFactory(ABC):

    @abstractmethod
    def get_service(self) -> FileHandler:
        pass

class S3Factory(FileHandlerAbstractFactory):

    def __init__(self):
        pass

    @lru_cache()
    def get_service(self) -> S3Handler:
        return S3Handler()

