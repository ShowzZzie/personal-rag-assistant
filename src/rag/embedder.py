from openai import OpenAI

from rag.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def get_embedding(text: str, model: str = settings.embedding_model) -> list[float]:

    if len(text.strip()) == 0:
        raise ValueError("Embedding string can't be empty")

    response = client.embeddings.create(input=text, model=model)

    return response.data[0].embedding


if __name__ == "__main__":
    get_embedding("yo")
