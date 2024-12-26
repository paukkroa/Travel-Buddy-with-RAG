from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


def load_documents(path) -> list[Document]:
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()

def split_documents(documents: list[Document] = [],
                    chunk_size = 500,
                    chunk_overlap = 100) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)