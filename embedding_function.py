import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer


def get_embedding_function(model="gemini"):
    if model == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True)
    elif model == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", show_progress_bar=True)
    else:
        embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    return embeddings