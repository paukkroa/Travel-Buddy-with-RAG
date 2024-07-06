import getpass
import os
from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    os.environ["OPENAI_API_KEY"] = getpass.getpass()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings