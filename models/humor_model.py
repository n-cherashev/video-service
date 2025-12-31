from asyncio import Protocol


class HumorModel(Protocol):
    async def score_segment(self, text: str, lang: str | None) -> float: ...
