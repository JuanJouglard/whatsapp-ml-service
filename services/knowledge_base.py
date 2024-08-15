from abc import ABC, abstractmethod

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import PreTrainedTokenizer


class KnowledgeBaseInterface(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def split_documents(self, chunk_size: int, tokenizer) -> list:
        pass


class LangchainKnowledgeBase(KnowledgeBaseInterface):

    def __init__(self, documents):
        self.processed_documents = [
                LangchainDocument(page_content=doc["clean_message"], metadata={"user": doc["user"], "date": doc["date"]}) for _, doc in documents.iterrows()
                ]
        self.documents = documents

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def split_documents(self, chunk_size: int, tokenizer: PreTrainedTokenizer):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=[" ", "."],
        )

        docs_processed = []

        for doc in self.processed_documents:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates

        unique_texts = {}
        docs_processed_unique = []

        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

