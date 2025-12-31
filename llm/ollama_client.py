from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from llm.base import LLMClient
from llm.prompts import (
    HUMOR_SYSTEM,
    BLOCK_ANALYSIS_SYSTEM,
    build_humor_user,
    build_messages,
    build_topics_user,
    build_block_analysis_user,
)
from llm.schemas import HUMOR_SCHEMA, TOPICS_SCHEMA, BLOCK_ANALYSIS_SCHEMA
from llm.settings import OllamaSettings


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp01(v: float) -> float:
    return max(0.0, min(float(v), 1.0))


def _safe_json_loads(s: str) -> Any | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        m = _JSON_RE.search(s)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _normalize_language(lang: Any) -> str | None:
    if not isinstance(lang, str):
        return None
    x = lang.strip().lower()
    if x in ("ru", "russian"):
        return "ru"
    if x in ("en", "english"):
        return "en"
    return x or None


def _normalize_slug(slug: str) -> str:
    s = (slug or "").strip().lower()
    if not s:
        return "general"
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^0-9a-zа-яё_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:48] or "general"


@dataclass
class _Cache:
    enabled: bool
    max_items: int
    items: dict[str, Any]

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self.items.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        if len(self.items) >= self.max_items:
            self.items.pop(next(iter(self.items)))
        self.items[key] = value


class OllamaLLMClient(LLMClient):
    def __init__(self, settings: OllamaSettings) -> None:
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for OllamaLLMClient. Install: pip install httpx")
        
        self.settings = settings
        self._client = httpx.Client(timeout=settings.timeout_seconds)
        self._cache = _Cache(settings.cache_enabled, settings.cache_max_items, {})

    def score_humor(self, text: str, language: str | None = None, metadata=None) -> float:
        if not self.settings.enabled:
            return 0.0

        lang = _normalize_language(language)
        text = (text or "").strip()
        if not text:
            return 0.0

        text = text[: self.settings.max_input_chars]

        user = build_humor_user(lang, text)
        messages = build_messages(HUMOR_SYSTEM, user)

        cache_key = self._cache_key("humor", lang, text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return float(cached)

        raw = self._chat(messages=messages, schema=HUMOR_SCHEMA)

        score = 0.0
        if isinstance(raw, dict):
            score = _clamp01(raw.get("score", 0.0))

        self._cache.set(cache_key, score)
        return score

    def summarize_topics(self, blocks: list[str], language: str | None = None) -> list[str]:
        if not self.settings.enabled:
            return ["general" for _ in blocks]

        lang = _normalize_language(language)
        safe_blocks = [(b or "").strip()[: self.settings.max_input_chars] for b in blocks]

        user = build_topics_user(lang, safe_blocks)
        messages = build_messages(TOPICS_SYSTEM, user)

        cache_key = self._cache_key("topics", lang, "\n\n".join(safe_blocks))
        cached = self._cache.get(cache_key)
        if isinstance(cached, list) and len(cached) == len(blocks):
            return [str(x) for x in cached]

        raw = self._chat(messages=messages, schema=TOPICS_SCHEMA)

        result = ["general" for _ in blocks]
        if isinstance(raw, dict) and isinstance(raw.get("topics"), list):
            items = raw["topics"]
            slugs: list[str] = []
            for it in items:
                if isinstance(it, dict):
                    slugs.append(_normalize_slug(str(it.get("slug") or "")))
            if slugs:
                result = self._fit_length(slugs, len(blocks))

        self._cache.set(cache_key, result)
        return result

    def _chat(self, messages: list[dict[str, Any]], schema: dict) -> Any | None:
        url = self.settings.base_url.rstrip("/") + "/api/chat"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "stream": False,
            "messages": messages,
            "format": schema,
            "options": self.settings.options,
        }
        if self.settings.keep_alive is not None:
            payload["keep_alive"] = self.settings.keep_alive

        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self.settings.retries:
            attempt += 1
            started = time.time()
            try:
                r = self._client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()

                content = None
                if isinstance(data, dict):
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")

                parsed = _safe_json_loads(str(content or ""))
                if parsed is not None:
                    return parsed

                return None
            except Exception as exc:
                last_exc = exc
                elapsed_ms = int((time.time() - started) * 1000)
                print(f"OllamaLLMClient: chat failed (attempt={attempt}, {elapsed_ms}ms): {exc}")

                if attempt > self.settings.retries:
                    break
                time.sleep(self.settings.backoff_seconds * (2 ** (attempt - 1)))

        print(f"OllamaLLMClient: giving up after retries, last error: {last_exc}")
        return None

    def _cache_key(self, kind: str, lang: str | None, text: str) -> str:
        src = f"{kind}|{lang or 'unknown'}|{text}"
        return hashlib.sha256(src.encode("utf-8")).hexdigest()

    def analyze_blocks(self, blocks: list[dict], language: str | None = None) -> list[dict]:
        """Анализирует блоки для получения topic + humor одним запросом."""
        if not self.settings.enabled or not blocks:
            return self._fallback_blocks(blocks)

        lang = _normalize_language(language)
        
        # Подготовка блоков с индексами и обрезкой текста
        safe_blocks = []
        for i, block in enumerate(blocks):
            text = (block.get("text") or "").strip()
            safe_blocks.append({
                "i": i,
                "start": block.get("start", 0),
                "end": block.get("end", 0),
                "text": text[:self.settings.max_input_chars]
            })

        # Кеширование
        cache_key = self._cache_key("blocks", lang, str(safe_blocks))
        cached = self._cache.get(cache_key)
        if isinstance(cached, list) and len(cached) == len(blocks):
            return cached

        # LLM запрос
        user = build_block_analysis_user(lang, safe_blocks)
        messages = build_messages(BLOCK_ANALYSIS_SYSTEM, user)
        
        start_time = time.time()
        raw = self._chat(messages=messages, schema=BLOCK_ANALYSIS_SCHEMA)
        elapsed = time.time() - start_time
        
        print(f"LLM block analysis: {len(blocks)} blocks, {elapsed:.2f}s")

        # Парсинг результата
        result = self._parse_block_analysis(raw, len(blocks))
        self._cache.set(cache_key, result)
        return result

    def _fallback_blocks(self, blocks: list[dict]) -> list[dict]:
        """Fallback результат при отключенном LLM."""
        return [
            {
                "topic_slug": "general",
                "topic_title": "General",
                "topic_confidence": 0.1,
                "humor_score": 0.0,
                "humor_label": "none"
            }
            for _ in blocks
        ]

    def _parse_block_analysis(self, raw: Any, expected_count: int) -> list[dict]:
        """Парсит результат block analysis с валидацией."""
        if not isinstance(raw, dict) or "items" not in raw:
            return self._fallback_blocks([{}] * expected_count)
        
        items = raw["items"]
        if not isinstance(items, list):
            return self._fallback_blocks([{}] * expected_count)
        
        # Сортируем по индексу и заполняем пропуски
        indexed_items = {}
        for item in items:
            if isinstance(item, dict) and "i" in item:
                idx = item["i"]
                if isinstance(idx, int) and 0 <= idx < expected_count:
                    topic_slug = _normalize_slug(str(item.get("topic_slug", "general")))
                    topic_confidence = _clamp01(item.get("topic_confidence", 0.1))
                    
                    # Enforce rule: if topic_slug="general" then confidence <= 0.35
                    if topic_slug == "general" and topic_confidence > 0.35:
                        topic_confidence = 0.35
                    
                    indexed_items[idx] = {
                        "topic_slug": topic_slug,
                        "topic_title": str(item.get("topic_title", "General"))[:80],
                        "topic_confidence": topic_confidence,
                        "humor_score": _clamp01(item.get("humor_score", 0.0)),
                        "humor_label": str(item.get("humor_label", "none"))
                    }
        
        # Заполняем результат с fallback для отсутствующих индексов
        result = []
        for i in range(expected_count):
            if i in indexed_items:
                result.append(indexed_items[i])
            else:
                result.append({
                    "topic_slug": "general",
                    "topic_title": "General", 
                    "topic_confidence": 0.1,
                    "humor_score": 0.0,
                    "humor_label": "none"
                })
        
        return result

    def refine_candidates(
        self, 
        candidates: list[dict], 
        language: str | None = None
    ) -> list[dict]:
        """Уточняет кандидатов через LLM анализ."""
        if not self.settings.enabled or not candidates:
            return self._fallback_candidates(candidates)
        
        lang = _normalize_language(language)
        
        # Подготовка данных
        safe_candidates = []
        for i, candidate in enumerate(candidates):
            text = (candidate.get("text") or "").strip()
            safe_candidates.append({
                "i": i,
                "start": candidate.get("start", 0),
                "end": candidate.get("end", 0),
                "text": text[:self.settings.max_input_chars],
                "reasons": candidate.get("reasons", []),
                "features": candidate.get("features", {})
            })
        
        # LLM запрос для уточнения
        user = self._build_refine_user(lang, safe_candidates)
        messages = build_messages("You are a video content analyst. Analyze candidate segments and provide detailed insights.", user)
        
        start_time = time.time()
        raw = self._chat(messages=messages, schema=self._get_refine_schema())
        elapsed = time.time() - start_time
        
        print(f"LLM candidate refine: {len(candidates)} candidates, {elapsed:.2f}s")
        
        # Парсинг результата
        result = self._parse_refine_result(raw, candidates)
        return result
    
    def _fallback_candidates(self, candidates: list[dict]) -> list[dict]:
        """Fallback для кандидатов при ошибке LLM."""
        return [
            {
                "start": c.get("start", 0),
                "end": c.get("end", 0),
                "title": f"Segment {i+1}",
                "description": "Content analysis unavailable",
                "why_interesting": "Based on automated signals",
                "tags": c.get("reasons", []),
                "humor_score": 0.0,
                "confidence": 0.1,
                "original_score": c.get("score", 0.5)
            }
            for i, c in enumerate(candidates)
        ]
    
    def _build_refine_user(self, lang: str | None, candidates: list[dict]) -> str:
        """Строит user prompt для уточнения кандидатов."""
        lang_note = "Respond in Russian." if lang == "ru" else "Respond in English."
        
        prompt = f"{lang_note}\n\nAnalyze these video segments and provide insights:\n\n"
        
        for c in candidates:
            prompt += f"Segment {c['i']+1} ({c['start']:.1f}s-{c['end']:.1f}s):\n"
            prompt += f"Reasons: {', '.join(c['reasons'])}\n"
            prompt += f"Text: {c['text'][:200]}...\n\n"
        
        return prompt
    
    def _get_refine_schema(self) -> dict:
        """JSON schema для уточнения кандидатов."""
        return {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "why_interesting": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "humor_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["i", "title", "description", "confidence"]
                    }
                }
            },
            "required": ["segments"]
        }
    
    def _parse_refine_result(self, raw: Any, original_candidates: list[dict]) -> list[dict]:
        """Парсит результат уточнения кандидатов."""
        if not isinstance(raw, dict) or "segments" not in raw:
            return self._fallback_candidates(original_candidates)
        
        segments = raw["segments"]
        if not isinstance(segments, list):
            return self._fallback_candidates(original_candidates)
        
        # Индексируем результаты
        indexed_results = {}
        for seg in segments:
            if isinstance(seg, dict) and "i" in seg:
                idx = seg["i"]
                if isinstance(idx, int) and 0 <= idx < len(original_candidates):
                    indexed_results[idx] = seg
        
        # Формируем финальный результат
        result = []
        for i, orig in enumerate(original_candidates):
            if i in indexed_results:
                refined = indexed_results[i]
                result.append({
                    "start": orig["start"],
                    "end": orig["end"],
                    "title": str(refined.get("title", f"Segment {i+1}"))[:100],
                    "description": str(refined.get("description", "No description"))[:300],
                    "why_interesting": str(refined.get("why_interesting", "Automated detection"))[:200],
                    "tags": refined.get("tags", orig.get("reasons", [])),
                    "humor_score": _clamp01(refined.get("humor_score", 0.0)),
                    "confidence": _clamp01(refined.get("confidence", 0.1)),
                    "original_score": orig.get("score", 0.5)
                })
            else:
                # Fallback для этого кандидата
                result.append({
                    "start": orig["start"],
                    "end": orig["end"],
                    "title": f"Segment {i+1}",
                    "description": "Content analysis unavailable",
                    "why_interesting": "Based on automated signals",
                    "tags": orig.get("reasons", []),
                    "humor_score": 0.0,
                    "confidence": 0.1,
                    "original_score": orig.get("score", 0.5)
                })
        
        return result