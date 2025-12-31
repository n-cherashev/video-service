from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chapter:
    start: float
    end: float
    title: str
    description: str