from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(
    text: str,
    model: str = "text-embedding-3-small"
) -> list[float]:

    if len(text.strip())==0:
        raise ValueError("Embedding string can't be empty")

    response = client.embeddings.create(
        input=text,
        model=model
        )
    
    return response.data[0].embedding


if __name__ == "__main__":
    get_embedding("yo")