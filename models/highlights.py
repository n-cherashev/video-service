from dataclasses import dataclass
from typing import Literal


HighlightType = Literal["top_interest", "rules"]


@dataclass(frozen=True, slots=True)
class Highlight:
    start: float
    end: float
    type: HighlightType
    score: float