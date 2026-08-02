import os

from anthropic import Anthropic
from dotenv import load_dotenv

from rag.schemas import Answer, RetrievedChunk

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def synthesize(
    user_question: str,
    source_chunks: list[RetrievedChunk],
    collection: str,
    model: str = "claude-haiku-4-5",
) -> Answer:

    source_lines: list[str] = []
    for i, chunk in enumerate(source_chunks):
        title = chunk.chunk.chunk_metadata.source_file
        text = chunk.chunk.text
        source_lines.append(f"[{i+1}] (source: {title}) \"{text}\"")

    sources = "\n".join(source_lines)

    message = client.messages.create(
        max_tokens=1024, # to be replaced later by Config value,
        messages=[
            {
                "role": "user",
                "content": f"Context: {sources}\n\nQuestion: {user_question}\n\n"
                "Answer using only the context above. Cite sources, e.g. [1], [2]"
            }
        ],
        model=model
    )

    assert message.content[0].type == "text"

    final_answer = Answer(
        question = user_question,
        answer = message.content[0].text,
        sources = source_chunks,
        model = model,
        input_tokens = message.usage.input_tokens,
        output_tokens = message.usage.output_tokens,
        collection = collection,
    )

    return final_answer