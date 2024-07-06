from langchain.vectorstores.chroma import Chroma
from langchain.schema.document import Document
from embedding_function import get_embedding_function
import os
import shutil
import hashlib

CHROMA_PATH = "chroma"

def create_chunk_ids(chunks) -> list[Document]:
    """
    Creates an unique md5 hash for each chunk. 
    Follows this logic: MD5("source:page:chunk_index")
    i.e. data/example.pdf:1:2 -> 'b8146f8577c4101ba8ce7308b0124ad8'
    
    params:
    chunks <list[Document]>: text chunks created using split_documents in data_loader

    return: chunks <list[Document]>: updated chunks object with chunk_id's added into the metadata
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk_id = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def create_or_update_chroma(chunks: list[Document]) -> None:
    """
    Create chroma db if not exists. Add new documents if there are any.

    params: 
    chunks <list[Document]>: text chunks created using split_documents in data_loader

    return: None
    """
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = create_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
