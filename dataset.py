from datasets import Dataset
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument


SEPARATORS = [".", ","]

class Chat():
    def __init__(self, source: String):
        self.chat_df = pd.read_pickle(source)


    def split_documents(
            chunk_size: int,
            knowledge_base: List[LangchainDocument],
            tokenizer: Tokenizer,
            ) -> List[LangchainDocument]:

        """

        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.

        """

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size / 10),
                add_start_index=True,
                strip_whitespace=True,
                separators=SEPARATORS,
                )

        docs_processed = []

        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates

        unique_texts = {}
        docs_processed_unique = []

        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique


