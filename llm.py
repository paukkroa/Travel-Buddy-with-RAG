from langchain_community.llms.ollama import Ollama
import google.generativeai as gemini
from openai import OpenAI
import os

def prompt_model(prompt, model_type, model_name):
    """
    Generic function to use different models for generating responses.
    Saves space from other files.
    """
    if model_type == "ollama":
        model = Ollama(model=model_name)
        response_text = model.invoke(prompt)

    elif model_type == "gemini":
        gemini.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = gemini.GenerativeModel(model_name)
        response_text = model.generate_content(prompt)

    elif model_type == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    organization=os.environ["OPENAI_ORG_ID"]) 
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        response_text = completion.choices[0].message.content.encode("utf-8").decode()

    else:
        raise Exception(f"Model {model_type}:{model_name} not supported.")
    
    return response_text