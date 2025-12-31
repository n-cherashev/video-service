"""
LLM Refinement Handler - использует GPT-4 для анализа топ-30 кандидатов.

Анализирует:
1. Hook (зацепка в первые 3 сек)
2. Эмоциональность
3. Завершённость мысли
4. Понятность без контекста
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from core.base_handler import BaseHandler

try:
    from openai import AsyncOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMRefineCandidatesHandler(BaseHandler):
    """LLM анализ для топ-N кандидатов с оценкой viral-потенциала.

    Использует GPT-4 для:
    - Оценки качества hook (первые 3 секунды)
    - Анализа эмоциональности
    - Проверки завершённости мысли
    - Оценки понятности без контекста
    """

    SYSTEM_PROMPT = """Ты эксперт по анализу видео-контента и выбору viral-моментов.

Твоя задача - оценить фрагменты видео по их viral-потенциалу.
Для каждого фрагмента оцени:

1. **Hook (0-1)**: Насколько первые 3 секунды захватывают внимание?
   - 0.9-1.0: Мощный hook, сразу цепляет
   - 0.6-0.8: Хороший hook, интересное начало
   - 0.3-0.5: Средний hook, требует внимания
   - 0-0.2: Слабый hook, может проскроллиться

2. **Emotion (0-1)**: Эмоциональная насыщенность
   - Высокая эмоция (радость, удивление, смех) = высокий балл
   - Нейтральное повествование = низкий балл

3. **Completeness (0-1)**: Завершённость мысли
   - Полная история/мысль/шутка = высокий балл
   - Обрывается на середине = низкий балл

4. **Standalone (0-1)**: Понятность без контекста
   - Понятно без просмотра всего видео = высокий балл
   - Требует контекста = низкий балл

Ответь в JSON формате:
```json
{
  "candidates": [
    {
      "index": 0,
      "hook_score": 0.8,
      "emotion_score": 0.7,
      "completeness_score": 0.9,
      "standalone_score": 0.6,
      "viral_potential": 0.75,
      "reasoning": "Краткое обоснование на русском"
    }
  ]
}
```"""

    USER_PROMPT_TEMPLATE = """Проанализируй следующие фрагменты видео на предмет viral-потенциала.

Транскрипт полного видео (для контекста):
{full_transcript}

Кандидаты на viral-моменты:
{candidates_text}

Оцени каждый кандидат по 4 критериям и рассчитай viral_potential как взвешенную сумму:
viral_potential = 0.35*hook + 0.25*emotion + 0.20*completeness + 0.20*standalone
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_candidates: int = 30,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        max_retries: int = 3,
        backoff_base_seconds: float = 2.0,
        batch_size: int = 10,
    ):
        self.api_key = api_key
        self.model = model
        self.max_candidates = max_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.batch_size = batch_size
        self._client: Optional[OpenAI] = None

    def _ensure_client(self) -> bool:
        """Инициализирует OpenAI клиент."""
        if not OPENAI_AVAILABLE:
            print("⚠️ OpenAI not installed. Run: pip install openai")
            return False

        if self._client is None:
            if not self.api_key:
                print("⚠️ OpenAI API key not provided")
                return False
            self._client = OpenAI(api_key=self.api_key)

        return True

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[LLM Refine] LLMRefineCandidatesHandler")

        # Получаем кандидатов (из viral_clips или candidates)
        candidates = context.get("viral_clips") or context.get("candidates", [])

        if not candidates:
            print("✓ LLM refine: no candidates")
            context["refined_candidates"] = []
            return context

        # Ограничиваем количество кандидатов
        candidates = candidates[:self.max_candidates]

        if not self._ensure_client():
            print("⚠️ LLM refine: using fallback (no API key)")
            context["refined_candidates"] = self._fallback_refine(candidates)
            return context

        transcript = context.get("full_transcript", "")
        transcript_segments = context.get("transcript_segments", [])

        try:
            refined = self._refine_with_llm(candidates, transcript, transcript_segments)
            context["refined_candidates"] = refined

            # Обновляем viral_clips с новыми скорами
            if context.get("viral_clips"):
                context["viral_clips"] = self._merge_llm_scores(
                    context["viral_clips"],
                    refined
                )

            print(f"✓ LLM refine: {len(refined)} candidates refined")

        except Exception as e:
            print(f"❌ LLM refine failed: {e}")
            context["refined_candidates"] = self._fallback_refine(candidates)

        return context

    def _refine_with_llm(
        self,
        candidates: List[Dict[str, Any]],
        full_transcript: str,
        transcript_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Отправляет кандидатов на анализ LLM."""
        all_refined = []

        # Обрабатываем батчами
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            # Формируем текст кандидатов
            candidates_text = self._format_candidates(batch, transcript_segments)

            # Готовим prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                full_transcript=full_transcript[:3000],  # Ограничиваем контекст
                candidates_text=candidates_text,
            )

            # Отправляем запрос с retry
            refined_batch = self._call_llm_with_retry(user_prompt, batch_num, len(batch))

            # Добавляем оригинальные данные
            for j, refined in enumerate(refined_batch):
                if j < len(batch):
                    refined["start"] = batch[j].get("start", 0)
                    refined["end"] = batch[j].get("end", 0)
                    refined["original_score"] = batch[j].get("score", 0)

            all_refined.extend(refined_batch)

        return all_refined

    def _format_candidates(
        self,
        candidates: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
    ) -> str:
        """Форматирует кандидатов для промпта."""
        lines = []

        for i, c in enumerate(candidates):
            start = c.get("start", 0)
            end = c.get("end", 0)
            score = c.get("score", 0)

            # Извлекаем текст для этого временного диапазона
            text = self._extract_text_for_range(start, end, transcript_segments)

            # Формируем описание
            score_breakdown = c.get("score_breakdown", {})
            breakdown_str = ", ".join(f"{k}={v:.2f}" for k, v in score_breakdown.items())

            lines.append(f"""
Кандидат #{i+1}:
- Время: {start:.1f}s - {end:.1f}s (длина: {end-start:.1f}s)
- Score: {score:.3f} ({breakdown_str})
- Текст: {text[:500]}
""")

        return "\n".join(lines)

    def _extract_text_for_range(
        self,
        start: float,
        end: float,
        segments: List[Dict[str, Any]],
    ) -> str:
        """Извлекает текст для временного диапазона."""
        texts = []
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", seg_start))

            # Проверяем пересечение
            if not (seg_end <= start or seg_start >= end):
                text = seg.get("text", "").strip()
                if text:
                    texts.append(text)

        return " ".join(texts) if texts else "[Нет текста]"

    def _call_llm_with_retry(
        self,
        user_prompt: str,
        batch_num: int,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Вызывает LLM с retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                    print(f"   Retry batch {batch_num} (attempt {attempt + 1}) after {delay}s...")
                    time.sleep(delay)

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                # Парсим ответ
                content = response.choices[0].message.content
                result = json.loads(content)

                candidates = result.get("candidates", [])
                print(f"   ✓ Batch {batch_num}: {len(candidates)} refined")

                return candidates

            except Exception as e:
                if attempt == self.max_retries:
                    print(f"   ❌ Batch {batch_num} failed after {self.max_retries + 1} attempts: {e}")
                    return self._fallback_batch(batch_size)

        return self._fallback_batch(batch_size)

    def _merge_llm_scores(
        self,
        original_clips: List[Dict[str, Any]],
        refined: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Объединяет оригинальные скоры с LLM скорами."""
        # Создаём маппинг по start/end
        refined_map = {
            (r.get("start", 0), r.get("end", 0)): r
            for r in refined
        }

        result = []
        for clip in original_clips:
            key = (clip.get("start", 0), clip.get("end", 0))
            merged = clip.copy()

            if key in refined_map:
                llm_data = refined_map[key]
                merged["llm_scores"] = {
                    "hook": llm_data.get("hook_score", 0.5),
                    "emotion": llm_data.get("emotion_score", 0.5),
                    "completeness": llm_data.get("completeness_score", 0.5),
                    "standalone": llm_data.get("standalone_score", 0.5),
                    "viral_potential": llm_data.get("viral_potential", 0.5),
                }
                merged["llm_reasoning"] = llm_data.get("reasoning", "")

                # Пересчитываем итоговый скор с учётом LLM
                original_score = merged.get("score", 0.5)
                llm_viral = merged["llm_scores"]["viral_potential"]

                # 60% оригинальный скор + 40% LLM viral_potential
                merged["final_score"] = 0.6 * original_score + 0.4 * llm_viral

            result.append(merged)

        # Пересортируем по final_score
        result.sort(key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)

        return result

    def _fallback_refine(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback когда LLM недоступен."""
        return [
            {
                "index": i,
                "start": c.get("start", 0),
                "end": c.get("end", 0),
                "hook_score": 0.5,
                "emotion_score": 0.5,
                "completeness_score": 0.5,
                "standalone_score": 0.5,
                "viral_potential": c.get("score", 0.5),
                "reasoning": "LLM analysis unavailable",
                "original_score": c.get("score", 0.5),
            }
            for i, c in enumerate(candidates)
        ]

    def _fallback_batch(self, size: int) -> List[Dict[str, Any]]:
        """Fallback для одного батча."""
        return [
            {
                "index": i,
                "hook_score": 0.5,
                "emotion_score": 0.5,
                "completeness_score": 0.5,
                "standalone_score": 0.5,
                "viral_potential": 0.5,
                "reasoning": "Analysis failed",
            }
            for i in range(size)
        ]
