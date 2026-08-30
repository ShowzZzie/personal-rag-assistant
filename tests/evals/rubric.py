from enum import Enum
from pydantic import BaseModel


RUBRIC = {
    "factual": "Is every claim in the answer supported by the retrieved chunks? YES / PARTIALLY / NO",
    "cited": "Does every claim carry a citation marker like [1] or [2]? YES / PARTIALLY / NO",
    "grounded": "Does the answer avoid stating anything not present in the retrieved chunks? YES / PARTIALLY / NO",
}

class Score(str, Enum):
    YES = "yes"
    PARTIALLY = "partially"
    NO = "no"

class JudgeScore(BaseModel):
    name: str
    score: Score
    reasoning: str