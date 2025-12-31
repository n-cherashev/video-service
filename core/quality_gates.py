"""
Quality Gates - проверки качества перед дорогими операциями.

Используется для:
1. Фильтрации кандидатов перед LLM refinement
2. Проверки данных перед refit
3. Валидации результатов перед сохранением
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from models.candidates import ScoredClipV2, ScoreComponents


@dataclass
class QualityGateResult:
    """Результат проверки quality gate."""
    passed: bool
    gate_name: str
    reason: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class QualityGateConfig:
    """Конфигурация quality gates."""
    # Минимальные пороги для LLM refinement
    min_hook_for_llm: float = 0.3
    min_clarity_for_llm: float = 0.25
    min_score_for_llm: float = 0.4

    # Минимальные пороги для refit
    min_score_for_refit: float = 0.35
    min_components_for_refit: int = 2  # Минимум 2 компонента > 0.5

    # Лимиты кандидатов
    max_candidates_for_llm: int = 15
    max_candidates_for_refit: int = 50

    # Минимальные требования к данным
    min_timeline_points: int = 10
    min_transcript_segments: int = 1


def check_llm_eligibility(
    clip: ScoredClipV2,
    config: Optional[QualityGateConfig] = None,
) -> QualityGateResult:
    """Проверяет, подходит ли клип для LLM refinement.

    Критерии:
    1. hook >= min_hook_for_llm
    2. clarity >= min_clarity_for_llm
    3. score >= min_score_for_llm
    """
    cfg = config or QualityGateConfig()

    # Hook check
    if clip.components.hook < cfg.min_hook_for_llm:
        return QualityGateResult(
            passed=False,
            gate_name="hook_threshold",
            reason=f"Hook {clip.components.hook:.2f} < {cfg.min_hook_for_llm}",
            value=clip.components.hook,
            threshold=cfg.min_hook_for_llm,
        )

    # Clarity check
    if clip.components.clarity < cfg.min_clarity_for_llm:
        return QualityGateResult(
            passed=False,
            gate_name="clarity_threshold",
            reason=f"Clarity {clip.components.clarity:.2f} < {cfg.min_clarity_for_llm}",
            value=clip.components.clarity,
            threshold=cfg.min_clarity_for_llm,
        )

    # Score check
    if clip.score < cfg.min_score_for_llm:
        return QualityGateResult(
            passed=False,
            gate_name="score_threshold",
            reason=f"Score {clip.score:.2f} < {cfg.min_score_for_llm}",
            value=clip.score,
            threshold=cfg.min_score_for_llm,
        )

    return QualityGateResult(
        passed=True,
        gate_name="llm_eligibility",
        reason="Clip meets LLM refinement criteria",
    )


def check_refit_eligibility(
    clip: ScoredClipV2,
    config: Optional[QualityGateConfig] = None,
) -> QualityGateResult:
    """Проверяет, подходит ли клип для refit.

    Критерии:
    1. score >= min_score_for_refit
    2. Минимум min_components_for_refit компонентов > 0.5
    """
    cfg = config or QualityGateConfig()

    # Score check
    if clip.score < cfg.min_score_for_refit:
        return QualityGateResult(
            passed=False,
            gate_name="refit_score",
            reason=f"Score {clip.score:.2f} < {cfg.min_score_for_refit}",
            value=clip.score,
            threshold=cfg.min_score_for_refit,
        )

    # Components check
    comp = clip.components
    good_components = sum(1 for v in [
        comp.hook, comp.pace, comp.clarity,
        comp.intensity, comp.emotion, comp.boundary
    ] if v > 0.5)

    if good_components < cfg.min_components_for_refit:
        return QualityGateResult(
            passed=False,
            gate_name="refit_components",
            reason=f"Only {good_components} components > 0.5 (need {cfg.min_components_for_refit})",
            value=float(good_components),
            threshold=float(cfg.min_components_for_refit),
        )

    return QualityGateResult(
        passed=True,
        gate_name="refit_eligibility",
        reason="Clip meets refit criteria",
    )


def filter_for_llm(
    clips: List[ScoredClipV2],
    config: Optional[QualityGateConfig] = None,
) -> tuple[List[ScoredClipV2], List[QualityGateResult]]:
    """Фильтрует клипы для LLM refinement.

    Returns:
        Tuple (eligible_clips, gate_results)
    """
    cfg = config or QualityGateConfig()

    eligible = []
    results = []

    for clip in clips:
        result = check_llm_eligibility(clip, cfg)
        results.append(result)
        if result.passed:
            eligible.append(clip)

    # Ограничиваем количество
    if len(eligible) > cfg.max_candidates_for_llm:
        # Берём топ по score
        eligible.sort(key=lambda c: c.score, reverse=True)
        eligible = eligible[:cfg.max_candidates_for_llm]

    return eligible, results


def filter_for_refit(
    clips: List[ScoredClipV2],
    config: Optional[QualityGateConfig] = None,
) -> tuple[List[ScoredClipV2], List[QualityGateResult]]:
    """Фильтрует клипы для refit.

    Returns:
        Tuple (eligible_clips, gate_results)
    """
    cfg = config or QualityGateConfig()

    eligible = []
    results = []

    for clip in clips:
        result = check_refit_eligibility(clip, cfg)
        results.append(result)
        if result.passed:
            eligible.append(clip)

    # Ограничиваем количество
    if len(eligible) > cfg.max_candidates_for_refit:
        eligible.sort(key=lambda c: c.score, reverse=True)
        eligible = eligible[:cfg.max_candidates_for_refit]

    return eligible, results


def check_data_quality(
    context: Dict[str, Any],
    config: Optional[QualityGateConfig] = None,
) -> List[QualityGateResult]:
    """Проверяет качество данных для анализа.

    Проверки:
    1. Достаточно timeline points
    2. Есть transcript segments
    3. Есть audio features
    """
    cfg = config or QualityGateConfig()
    results = []

    # Timeline check
    timeline = context.get("timeline_points", []) or context.get("timeline", [])
    if len(timeline) < cfg.min_timeline_points:
        results.append(QualityGateResult(
            passed=False,
            gate_name="timeline_data",
            reason=f"Only {len(timeline)} timeline points (need {cfg.min_timeline_points})",
            value=float(len(timeline)),
            threshold=float(cfg.min_timeline_points),
        ))
    else:
        results.append(QualityGateResult(
            passed=True,
            gate_name="timeline_data",
            reason=f"{len(timeline)} timeline points available",
        ))

    # Transcript check
    segments = context.get("transcript_segments", [])
    if len(segments) < cfg.min_transcript_segments:
        results.append(QualityGateResult(
            passed=False,
            gate_name="transcript_data",
            reason=f"Only {len(segments)} transcript segments",
            value=float(len(segments)),
            threshold=float(cfg.min_transcript_segments),
        ))
    else:
        results.append(QualityGateResult(
            passed=True,
            gate_name="transcript_data",
            reason=f"{len(segments)} transcript segments available",
        ))

    # Audio features check
    audio_features = context.get("audio_features", {})
    if not audio_features or not audio_features.get("loudness"):
        results.append(QualityGateResult(
            passed=False,
            gate_name="audio_data",
            reason="No audio features available",
        ))
    else:
        results.append(QualityGateResult(
            passed=True,
            gate_name="audio_data",
            reason="Audio features available",
        ))

    return results


def summarize_gates(results: List[QualityGateResult]) -> Dict[str, Any]:
    """Создаёт сводку по quality gates."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(results) if results else 0.0,
        "failed_gates": [r.gate_name for r in failed],
        "details": [
            {
                "gate": r.gate_name,
                "passed": r.passed,
                "reason": r.reason,
            }
            for r in results
        ],
    }
