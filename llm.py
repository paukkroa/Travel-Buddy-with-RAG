import ollama
import google.generativeai as gemini
from openai import OpenAI
import os

def prompt_model(prompt, 
                 model_type="gemini", 
                 model_name="gemini-2.0-flash-exp", 
                 sys_prompt="You are a helpful travel assistant. Do not use highlighted or bolded words (words like **title**), just use plain text.") -> str:
    """
    Generic function to use different models for generating responses.
    Saves space from other files.
    """
    if model_type == "ollama":
        response = ollama.chat(model=model_name, messages=[
            {
            'role': 'system',
            'content': sys_prompt,
            },
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        response_text = response['message']['content']

    elif model_type == "gemini":
        gemini.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = gemini.GenerativeModel(model_name,
                                       system_instruction=[sys_prompt])
        response = model.generate_content(prompt)
        response_text = response.candidates[0].content.parts[0].text

    elif model_type == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    organization=os.environ["OPENAI_ORG_ID"]) 
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                'role': 'system',
                'content': sys_prompt,
                },
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