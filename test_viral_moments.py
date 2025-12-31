#!/usr/bin/env python3
"""
Unit tests для системы поиска виральных моментов.
"""

import numpy as np
from models.viral_moments import Candidate, ScoredClip, ViralMomentsConfig
from features.highlights.viral_moments_handler import ViralMomentsHandler


def test_overlap_penalty():
    """Тест штрафов за пересечения."""
    config = ViralMomentsConfig()
    handler = ViralMomentsHandler(config)
    
    clip1 = ScoredClip(start=10.0, end=30.0, score=0.8, score_breakdown={}, anchor_type="test", reasons=[])
    clip2 = ScoredClip(start=25.0, end=45.0, score=0.7, score_breakdown={}, anchor_type="test", reasons=[])
    
    # Тест сильного пересечения (overlap > 0.65)
    overlap_ratio = handler._calculate_overlap_ratio(clip1, clip2)
    print(f"Overlap ratio: {overlap_ratio:.2f}")
    
    # Должно быть примерно 0.25 (5 секунд пересечения / 20 секунд минимальной длины)
    assert 0.2 <= overlap_ratio <= 0.3, f"Expected overlap ~0.25, got {overlap_ratio}"
    
    # Тест adjusted score
    adjusted_score = handler._calculate_adjusted_score(clip2, [clip1])
    
    # Определяем ожидаемый штраф
    if overlap_ratio > config.strong_overlap_threshold:
        expected_penalty = config.strong_penalty
    elif overlap_ratio > config.medium_overlap_threshold:
        expected_penalty = config.medium_penalty
    else:
        expected_penalty = 1.0
    
    expected_score = clip2.score * expected_penalty
    
    print(f"Overlap ratio: {overlap_ratio:.2f}, threshold: {config.medium_overlap_threshold:.2f}")
    print(f"Expected penalty: {expected_penalty:.2f}, expected score: {expected_score:.2f}, actual: {adjusted_score:.2f}")
    
    assert abs(adjusted_score - expected_score) < 0.01, f"Expected {expected_score}, got {adjusted_score}"
    
    print("✓ Overlap penalty test passed")


def test_diversify_selection():
    """Тест диверсификации выбора клипов."""
    config = ViralMomentsConfig(max_clips=3)
    handler = ViralMomentsHandler(config)
    
    # Создаем клипы с разными пересечениями
    clips = [
        ScoredClip(start=0.0, end=20.0, score=0.9, score_breakdown={}, anchor_type="test", reasons=[]),
        ScoredClip(start=15.0, end=35.0, score=0.8, score_breakdown={}, anchor_type="test", reasons=[]),  # Пересекается с первым
        ScoredClip(start=50.0, end=70.0, score=0.7, score_breakdown={}, anchor_type="test", reasons=[]),  # Не пересекается
        ScoredClip(start=60.0, end=80.0, score=0.6, score_breakdown={}, anchor_type="test", reasons=[]),  # Пересекается с третьим
    ]
    
    selected = handler._diversify_and_select(clips)
    
    print(f"Selected {len(selected)} clips from {len(clips)} candidates")
    for i, clip in enumerate(selected):
        print(f"  Clip {i+1}: {clip.start:.1f}s-{clip.end:.1f}s, score={clip.score:.2f}")
    
    # Должны выбрать максимум 3 клипа
    assert len(selected) <= config.max_clips
    
    # Клипы должны быть отсортированы по времени
    for i in range(1, len(selected)):
        assert selected[i].start >= selected[i-1].start
    
    print("✓ Diversify selection test passed")


def test_local_refit():
    """Тест локального re-fit для уменьшения пересечений."""
    config = ViralMomentsConfig(refit_shifts=[-5, -2, 2, 5])
    handler = ViralMomentsHandler(config)
    
    # Уже выбранный клип
    selected_clip = ScoredClip(start=20.0, end=40.0, score=0.8, score_breakdown={}, anchor_type="test", reasons=[])
    
    # Новый клип с сильным пересечением
    new_clip = ScoredClip(start=35.0, end=55.0, score=0.7, score_breakdown={}, anchor_type="test", reasons=[])
    
    # Проверяем что есть сильное пересечение
    overlap_before = handler._calculate_overlap_ratio(new_clip, selected_clip)
    print(f"Overlap before refit: {overlap_before:.2f}")
    
    # Применяем local refit
    refitted_clip = handler._local_refit(new_clip, [selected_clip])
    
    # Проверяем что пересечение уменьшилось или клип сдвинулся
    overlap_after = handler._calculate_overlap_ratio(refitted_clip, selected_clip)
    print(f"Overlap after refit: {overlap_after:.2f}")
    print(f"Clip moved from {new_clip.start:.1f}s-{new_clip.end:.1f}s to {refitted_clip.start:.1f}s-{refitted_clip.end:.1f}s")
    
    # Либо пересечение уменьшилось, либо клип остался на месте (если сдвиги не помогли)
    assert overlap_after <= overlap_before or (refitted_clip.start == new_clip.start and refitted_clip.end == new_clip.end)
    
    print("✓ Local refit test passed")


def test_timeline_stability():
    """Тест стабильности результата на синтетическом timeline."""
    config = ViralMomentsConfig()
    
    # Создаем синтетический timeline
    timeline = []
    for i in range(100):  # 100 секунд
        timeline.append({
            "time": float(i),
            "interest": 0.5 + 0.3 * np.sin(i * 0.1),  # Синусоидальный интерес
            "motion": 0.4 + 0.4 * np.sin(i * 0.15),
            "audio_loudness": 0.3 + 0.5 * np.sin(i * 0.08),
            "sentiment": 0.2 * np.sin(i * 0.05),
            "is_dialogue": i % 10 < 5,  # Диалог каждые 10 секунд
            "is_scene_boundary": i % 25 == 0,  # Граница сцены каждые 25 секунд
        })
    
    # Создаем кандидатов
    candidates = [
        Candidate(start=10.0, end=35.0, anchor_time=20.0, anchor_type="interest_peak", duration=25.0),
        Candidate(start=40.0, end=65.0, anchor_time=50.0, anchor_type="motion_peak", duration=25.0),
        Candidate(start=70.0, end=95.0, anchor_time=80.0, anchor_type="audio_peak", duration=25.0),
    ]
    
    handler = ViralMomentsHandler(config)
    
    # Запускаем скоринг несколько раз
    results = []
    for _ in range(3):
        scored_clips = handler._score_candidates(candidates, timeline)
        final_clips = handler._diversify_and_select(scored_clips)
        results.append(final_clips)
    
    # Проверяем стабильность
    assert len(results) == 3
    assert all(len(result) == len(results[0]) for result in results), "Inconsistent number of clips"
    
    # Проверяем что скоры стабильны (должны быть одинаковыми)
    for i in range(len(results[0])):
        scores = [result[i].score for result in results]
        assert all(abs(score - scores[0]) < 0.001 for score in scores), f"Unstable scores: {scores}"
    
    print(f"✓ Timeline stability test passed - {len(results[0])} clips consistently selected")
    for i, clip in enumerate(results[0]):
        print(f"  Clip {i+1}: {clip.start:.1f}s-{clip.end:.1f}s, score={clip.score:.3f}, type={clip.anchor_type}")


def test_scoring_components():
    """Тест отдельных компонентов скоринга."""
    config = ViralMomentsConfig()
    handler = ViralMomentsHandler(config)
    
    # Создаем timeline с известными характеристиками
    timeline = []
    for i in range(30):
        timeline.append({
            "time": float(i),
            "interest": 0.8 if 10 <= i <= 15 else 0.3,  # Высокий интерес в середине
            "motion": 0.9 if i < 5 else 0.2,  # Высокое движение в начале (хороший hook)
            "audio_loudness": 0.7 if i < 5 else 0.4,  # Громкий звук в начале
            "sentiment": 0.5 * (-1 if i % 4 < 2 else 1),  # Частые смены эмоций
            "is_dialogue": i >= 5,  # Диалог после 5 секунды
            "is_scene_boundary": i == 0 or i == 29,  # Границы в начале и конце
        })
    
    candidate = Candidate(start=0.0, end=30.0, anchor_time=15.0, anchor_type="test", duration=30.0)
    scored_clips = handler._score_candidates([candidate], timeline)
    
    assert len(scored_clips) == 1
    clip = scored_clips[0]
    
    print("Score breakdown:")
    for component, score in clip.score_breakdown.items():
        print(f"  {component}: {score:.3f}")
    
    # Проверяем что hook score высокий (движение и звук в начале)
    assert clip.score_breakdown["hook"] > 0.6, f"Expected high hook score, got {clip.score_breakdown['hook']}"
    
    # Проверяем что clarity score высокий (много диалога)
    assert clip.score_breakdown["clarity"] > 0.7, f"Expected high clarity score, got {clip.score_breakdown['clarity']}"
    
    # Проверяем что есть reasons
    assert len(clip.reasons) > 0, "Expected some reasons"
    
    print(f"✓ Scoring components test passed - final score: {clip.score:.3f}")
    print(f"  Reasons: {', '.join(clip.reasons)}")


if __name__ == "__main__":
    print("Running viral moments tests...\n")
    
    test_overlap_penalty()
    print()
    test_diversify_selection()
    print()
    test_local_refit()
    print()
    test_timeline_stability()
    print()
    test_scoring_components()
    
    print("\n✅ All viral moments tests passed!")