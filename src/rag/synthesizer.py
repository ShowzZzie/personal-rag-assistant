from anthropic import Anthropic

from rag import config
from rag.config import settings
from rag.schemas import Answer, RetrievedChunk

client = Anthropic(api_key=settings.anthropic_api_key)

def synthesize(
    user_question: str,
    source_chunks: list[RetrievedChunk],
    collection: str,
    model: str = settings.synthesis_model,
) -> Answer:

    source_lines: list[str] = []
    for i, chunk in enumerate(source_chunks):
        title = chunk.chunk.chunk_metadata.source_file
        text = chunk.chunk.text
        source_lines.append(f"[{i+1}] (source: {title}) \"{text}\"")

    sources = "\n".join(source_lines)

    message = client.messages.create(
        max_tokens=settings.synthesis_max_tokens,
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

    pricing = config.MODEL_PRICING.get(model)
    if pricing is None:
        cost = None
    else:
        input_price, output_price = pricing
        cost = (
            message.usage.input_tokens * input_price / 1_000_000
            + message.usage.output_tokens * output_price / 1_000_000
        )

    final_answer = Answer(
        question = user_question,
        answer = message.content[0].text,
        sources = source_chunks,
        model = model,
        input_tokens = message.usage.input_tokens,
        output_tokens = message.usage.output_tokens,
        cost_usd = cost,
        collection = collection,
    )

    return final_answer