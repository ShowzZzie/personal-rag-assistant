from rag.retriever import retrieve
from rag.store import chroma_persistent_client
import pytest

GOLDEN_PAIRS: list[dict[str,str]] = [
    {
        "question": "How does sleep impact mental health?",
        "document": "Improving sleep quality leads to better mental health- A meta-analysis of randomised controlled trials.pdf",
        "expected_chunk_contains": "sleep is causally related"
    },
    {
        "question": "Does it make any difference whether I implement sleep improvements with or without specialist supervision?",
        "document": "Improving sleep quality leads to better mental health- A meta-analysis of randomised controlled trials.pdf",
        "expected_chunk_contains": "face-to-face by a clinician"
    },
    {
        "question": "Will just sleeping better get rid of my burnout?",
        "document": "Improving sleep quality leads to better mental health- A meta-analysis of randomised controlled trials.pdf",
        "expected_chunk_contains": "almost zero effect"
    },
    {
        "question": "Are smart watches actually good at identifying stages of sleep?",
        "document": "A Validation of Six Wearable Devices for Estimating Sleep, Heart Rate and Heart Rate Variability in Healthy Adults.pdf",
        "expected_chunk_contains": "require\xa0improvement"
    },
    {
        "question": "If I wanted to check my sleep quality, what is the best method?",
        "document": "Measuring Subjective Sleep Quality- A Review.pdf",
        "expected_chunk_contains": "Pittsburgh Sleep Quality Index (PSQI)"
    },
    {
        "question": "what's the most commonly used method for sleep assesment?",
        "document": "Measuring Subjective Sleep Quality- A Review.pdf",
        "expected_chunk_contains": "sleep diary is the most widely-used"
    },
    {
        "question": "Should I drink alcohol to help me fall asleep?",
        "document": "Sleep Quality- A Narrative Review on Nutrition, Stimulants, and Physical Activity as Important Factors.pdf",
        "expected_chunk_contains": "Alcohol is not recommended before going to sleep"
    },
    {
        "question": "what's the ideal timing for the last meal before sleeping?",
        "document": "Sleep Quality- A Narrative Review on Nutrition, Stimulants, and Physical Activity as Important Factors.pdf",
        "expected_chunk_contains": "four hours before bedtime"
    },
    {
        "question": "how much sleep should I be getting daily to be in peak form?",
        "document": "Sleep is essential to health- an American Academy of Sleep Medicine position statement.pdf",
        "expected_chunk_contains": "7 or more hours per night"
    },
    {
        "question": "would you recommend exercise and activity for improving sleep?",
        "document": "The Effect of Physical Activity on Sleep Quality and Sleep Disorder- A Systematic Review.pdf",
        "expected_chunk_contains": "Regular physical activity can lead to improved sleep quality"
    },
    {
        "question": "would you recommend working out before sleep?",
        "document": "The Effect of Physical Activity on Sleep Quality and Sleep Disorder- A Systematic Review.pdf",
        "expected_chunk_contains": "especially in the evening or close to bedtime"
    },
    {
        "question": "can sleep deprivation have lethal consequences?",
        "document": "Role of sleep deprivation in immune-related disease risk and outcomes.pdf",
        "expected_chunk_contains": "5 of the top 15 leading causes of death"
    },
    {
        "question": "can short sleep make me more obese?",
        "document": "Role of sleep deprivation in immune-related disease risk and outcomes.pdf",
        "expected_chunk_contains": "about 55% higher risk"
    },
    {
        "question": "what can contribute to chronic sleep loss?",
        "document": "Role of sleep deprivation in immune-related disease risk and outcomes.pdf",
        "expected_chunk_contains": "smartphone addiction"
    },
]


def find_targets(phrase: str) -> set[tuple[str, int]]:
    """(document_id, chunk_index) pairs for every chunk containing this phrase."""
    col = chroma_persistent_client.get_collection("sleep")
    recs = col.get()
    return {
        (meta["document_id"], meta["chunk_index"])
        for doc, meta in zip(recs["documents"], recs["metadatas"])
        if phrase in doc
    }


@pytest.mark.evals
def test_ret_evals():
    hits = 0

    for pair in GOLDEN_PAIRS:
        results = retrieve(pair["question"],"sleep",3)
        targets = find_targets(pair["expected_chunk_contains"])
        found = any(
            (r.chunk.document_id, idx) in targets
            for r in results
            for idx in (r.chunk.chunk_index - 1, r.chunk.chunk_index, r.chunk.chunk_index + 1)
        )
        if found:
            hits += 1
        print(f"[{found}] {pair['question']}")

    recall = hits / len(GOLDEN_PAIRS)
    print(f"Recall@3: {recall:.2f} ({hits}/{len(GOLDEN_PAIRS)})")
    assert recall >= 0.70