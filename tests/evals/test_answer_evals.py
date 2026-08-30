import os

from anthropic import Anthropic
from dotenv import load_dotenv
from tests.evals.rubric import RUBRIC, JudgeScore, Score
from rag.query import query
from tests.evals.test_retrieval_evals import GOLDEN_PAIRS

load_dotenv()
judge_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

JUDGE_MODEL = "claude-opus-5"


def judge_answer(question: str, chunks: str, answer: str) -> list[JudgeScore]:
    criteria = "\n".join(f"- {name}: {desc}" for name, desc in RUBRIC.items())

    prompt = (
        f"You are evaluating a RAG system's answer.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"Evaluate against these criteria:\n{criteria}\n\n"
    )

    tool = {
        "name": "submit_scores",
        "description": "Submit rubric scores for the answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": list(RUBRIC.keys())},
                            "score": {"type": "string", "enum": ["yes", "partially", "no"]},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["name", "score", "reasoning"],
                    },
                }
            },
            "required": ["scores"],
        },
    }

    message = judge_client.messages.create(
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        model=JUDGE_MODEL,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_scores"},
    )

    blocks = [b for b in message.content if b.type == "tool_use"]
    assert blocks
    return [JudgeScore(**item) for item in blocks[0].input["scores"]]


def test_answer_evals():
    results: list[JudgeScore] = []

    for pair in GOLDEN_PAIRS[:10]:
        answer = query(pair["question"], "sleep")
        chunks = "\n\n".join(
            f"[{i+1}] {r.chunk.text}" for i, r in enumerate(answer.sources)
        )
        scores = judge_answer(pair["question"], chunks, answer.answer)
        results.extend(scores)

        for s in scores:
            print(f"  {s.name}: {s.score.value} — {s.reasoning}")
        print(f"[{pair['question']}]")

    yes = sum(1 for s in results if s.score == Score.YES)
    total = len(results)
    print(f"\nAnswer eval: {yes}/{total} = {yes/total:.2f} YES")
    assert yes / total >= 0.70