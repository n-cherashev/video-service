from __future__ import annotations

from typing import Any


HUMOR_SYSTEM = """You are a strict JSON-only classifier for short video segments.

Return ONLY valid JSON that matches the provided JSON Schema. No markdown, no comments.

Primary goal:
Estimate comedic / humorous intensity in the segment.

You may receive extra signals (optional):
- haslaughter: boolean
- laughterprob: number in [0,1]
- hasloudsound: boolean
Use them as evidence. If haslaughter=true or laughterprob>=0.6, humor is at least "light"
unless the segment is clearly non-comedic (e.g., screams, crying, alarms).

Hard rules:
- Always output fields: score, label, reason_codes.
- score must be a number (not string) clamped to [0,1].
- reason_codes MUST contain 1-3 unique items.
- If text is empty / only punctuation / unintelligible => score=0, label="none", reason_codes=["unknown_unclear"].
- If score >= 0.15 you MUST NOT use only "unknown_unclear" unless text is very noisy; prefer a best-guess reason.

Scoring anchors:
- 0.00–0.14 => none
- 0.15–0.34 => light
- 0.35–0.64 => likely
- 0.65–0.84 => clear
- 0.85–1.00 => strong

Reason code hints (choose best 1-3):
- reaction_laughter_implied: laughter/cheering/applause signal present
- irony_sarcasm: contradiction, mock-serious tone
- teasing_banter: playful insults/banter
- absurdity: nonsense / surreal logic / ridiculous situation
- wordplay: puns, unusual phrasing (language-agnostic)
- explicit_joke / funny_story / self_deprecation: only if clearly indicated
"""

BLOCK_ANALYSIS_SYSTEM = """You are a strict JSON-only analyzer for transcript blocks.

Return ONLY valid JSON matching the provided JSON Schema. No markdown, no extra text.
Return exactly N items (one per input block). Each item MUST contain the same i as the block index.

Inputs:
Each block contains transcript text and may include extra features:
- start/end seconds
- haslaughter / laughterprob
- hasloudsound
- speech_prob (optional)
Use features as weak evidence; transcript text is primary.

Topic rules:
- topic_slug: lowercase snake_case (1-4 words). Cyrillic allowed for Russian.
- If text is empty/garbled => topic_slug="general", topic_title="General" (or "Общее"), topic_confidence <= 0.30
- If block clearly indicates silence/no speech => topic_slug="silence", topic_title="Silence"/"Тишина", topic_confidence <= 0.30
- If topic_slug="general" then topic_confidence MUST be <= 0.35
- If topic_confidence >= 0.70 then topic_slug MUST NOT be "general" or "silence"

Humor rules:
- humor_score in [0,1], humor_label mapped by thresholds:
  <0.15 none, 0.15-0.35 light, 0.35-0.65 likely, 0.65-0.85 clear, >=0.85 strong
- If haslaughter=true or laughterprob>=0.6 then humor_score >= 0.15 unless the content is clearly non-comedic.

Quality rules:
- Prefer conservative confidence when transcript is noisy.
- Do not hallucinate specific named entities if they are not present in the text.
"""


def build_block_analysis_user(language: str | None, blocks: list[dict]) -> str:
    lang = language or "unknown"
    parts = [
        f"Detected language (may be wrong): {lang}",
        f"Number of blocks: {len(blocks)}",
        "",
        "Blocks (text may be noisy):",
    ]

    for b in blocks:
        i = b.get("i", 0)
        start = float(b.get("start", 0))
        end = float(b.get("end", 0))
        text = (b.get("text") or "").strip()

        # новые поля (если есть после YAMNet/FusionTimeline)
        haslaughter = b.get("haslaughter")
        laughterprob = b.get("laughterprob")
        hasloudsound = b.get("hasloudsound")

        parts.append(f"[BLOCK {i}] start={start:.1f}s end={end:.1f}s")
        parts.append(f"Signals: haslaughter={haslaughter} laughterprob={laughterprob} hasloudsound={hasloudsound}")
        parts.append(f'Text:\n"""\n{text}\n"""\n')

    return "\n".join(parts)


def build_humor_user(language: str | None, text: str) -> str:
    lang = language or "unknown"
    return f'Detected language (may be wrong): {lang}\n\nSegment:\n"""\n{text}\n"""'


def build_topics_user(language: str | None, blocks: list[str]) -> str:
    lang = language or "unknown"
    parts: list[str] = [f"Overall detected language (may be wrong): {lang}", f"Number of blocks: {len(blocks)}", "", "Blocks:"]
    for i, b in enumerate(blocks, 1):
        parts.append(f"[BLOCK {i}]\n\"\"\"\n{b}\n\"\"\"\n")
    return "\n".join(parts)


def build_messages(system: str, user: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]