#!/usr/bin/env python3
"""
Интеграционный тест для системы виральных моментов.
"""

from models.viral_moments import ViralMomentsConfig
from features.fusion.candidate_selection_handler import CandidateSelectionHandler
from features.highlights.viral_moments_handler import ViralMomentsHandler


def test_viral_moments_integration():
    """Тест полного пайплайна виральных моментов."""
    
    # Создаем синтетические данные
    context = {
        "duration_seconds": 120.0,
        "timeline": []
    }
    
    # Генерируем timeline с интересными моментами
    for i in range(120):
        context["timeline"].append({
            "time": float(i),
            "interest": 0.3 + 0.4 * (1 if 20 <= i <= 25 or 60 <= i <= 70 else 0),  # Пики интереса
            "motion": 0.2 + 0.6 * (1 if i < 10 or 80 <= i <= 90 else 0),  # Движение в начале и конце
            "audio_loudness": 0.4 + 0.5 * (1 if 30 <= i <= 35 else 0),  # Громкий звук
            "sentiment": 0.1 * (i % 10 - 5),  # Переменчивые эмоции
            "is_dialogue": i % 15 < 10,  # Диалог 2/3 времени
            "is_scene_boundary": i % 30 == 0,  # Границы сцен каждые 30 сек
            "has_loud_sound": 30 <= i <= 35,  # Громкие звуки
        })
    
    config = ViralMomentsConfig(max_clips=5)
    
    # Этап 1: Генерация кандидатов
    candidate_handler = CandidateSelectionHandler(config)
    context = candidate_handler.handle(context)
    
    candidates = context.get("viral_candidates", [])
    print(f"Generated {len(candidates)} candidates")
    
    assert len(candidates) > 0, "Should generate some candidates"
    
    # Этап 2: Скоринг и выбор виральных моментов
    viral_handler = ViralMomentsHandler(config)
    context = viral_handler.handle(context)
    
    viral_clips = context.get("viral_clips", [])
    highlights = context.get("highlights", [])
    
    print(f"Selected {len(viral_clips)} viral clips")
    
    # Проверяем результаты
    assert len(viral_clips) > 0, "Should select some viral clips"
    assert len(viral_clips) <= config.max_clips, f"Should not exceed max_clips ({config.max_clips})"
    assert len(highlights) == len(viral_clips), "Should have backward compatibility"
    
    # Проверяем структуру результатов
    for i, clip in enumerate(viral_clips):
        print(f"Clip {i+1}: {clip['start']:.1f}s-{clip['end']:.1f}s, score={clip['score']:.3f}")
        print(f"  Anchor: {clip['anchor_type']}, Reasons: {', '.join(clip['reasons'])}")
        
        # Проверяем обязательные поля
        assert "start" in clip
        assert "end" in clip
        assert "score" in clip
        assert "score_breakdown" in clip
        assert "anchor_type" in clip
        assert "reasons" in clip
        
        # Проверяем score_breakdown
        breakdown = clip["score_breakdown"]
        expected_components = ["hook", "pace", "clarity", "intensity", "emotion", "boundary"]
        for component in expected_components:
            assert component in breakdown, f"Missing component: {component}"
            assert 0 <= breakdown[component] <= 1, f"Component {component} out of range: {breakdown[component]}"
    
    # Проверяем что клипы отсортированы по времени
    for i in range(1, len(viral_clips)):
        assert viral_clips[i]["start"] >= viral_clips[i-1]["start"], "Clips should be sorted by start time"
    
    print("✓ Viral moments integration test passed")
    return True


def test_config_parameters():
    """Тест различных конфигураций."""
    
    # Тест с разными параметрами
    configs = [
        ViralMomentsConfig(max_clips=3, strong_overlap_threshold=0.8),
        ViralMomentsConfig(max_clips=10, candidate_durations=[20.0, 40.0]),
        ViralMomentsConfig(hook_weight=0.5, pace_weight=0.3),
    ]
    
    for i, config in enumerate(configs):
        print(f"Testing config {i+1}: max_clips={config.max_clips}")
        
        # Простой контекст
        context = {
            "duration_seconds": 60.0,
            "timeline": [
                {
                    "time": float(j),
                    "interest": 0.5,
                    "motion": 0.4,
                    "audio_loudness": 0.3,
                    "sentiment": 0.0,
                    "is_dialogue": True,
                    "is_scene_boundary": j == 0,
                    "has_loud_sound": False,
                }
                for j in range(60)
            ]
        }
        
        # Запускаем пайплайн
        candidate_handler = CandidateSelectionHandler(config)
        viral_handler = ViralMomentsHandler(config)
        
        context = candidate_handler.handle(context)
        context = viral_handler.handle(context)
        
        viral_clips = context.get("viral_clips", [])
        assert len(viral_clips) <= config.max_clips, f"Config {i+1}: exceeded max_clips"
        
        print(f"  Generated {len(viral_clips)} clips")
    
    print("✓ Config parameters test passed")
    return True


if __name__ == "__main__":
    print("Testing viral moments integration...\n")
    
    success = True
    success &= test_viral_moments_integration()
    print()
    success &= test_config_parameters()
    
    if success:
        print("\n✅ All viral moments integration tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)