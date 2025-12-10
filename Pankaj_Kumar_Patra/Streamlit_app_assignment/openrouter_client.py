import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def ask_openrouter(messages):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")  # MUST NOT be None
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message["content"]
