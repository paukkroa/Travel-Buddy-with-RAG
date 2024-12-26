from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from embedding_function import get_embedding_function
from create_or_update_db import CHROMA_PATH
from llm import prompt_model

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_only(query_text: str):
    """
    Get relevant context from database based on the query text.
    """
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Create the context text.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    return results, context_text

def query_and_response(query_text: str,
          model_type = "gemini",
          model_name = "gemini-2.0-flash-exp"):
    """
    Creates formatted response based on the query text.
    Performs a RAG search on the database and returns a response based on the context and query text.
    """
    # Get text only context for the query
    results, context_text = query_only(query_text)

    # Prepare the prompt.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Get the response from the specific model.
    response_text = prompt_model(prompt, model_type, model_name)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text