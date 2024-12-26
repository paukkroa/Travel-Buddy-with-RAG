from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from client import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(path) -> list[Document]:
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()

def split_documents(documents: list[Document]) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def main():
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)

if __name__ == "__main__":
    main()