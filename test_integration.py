#!/usr/bin/env python3
"""
Интеграционный тест нового пайплайна без внешних ML зависимостей.
"""

import json
from config.settings import VideoServiceSettings


def test_pipeline_with_new_features():
    """Тест что пайплайн собирается с новыми feature flags."""
    
    # Настройки с включенными новыми фичами
    settings = VideoServiceSettings(
        input_video_path="/tmp/test.mp4",
        enable_yamnet=True,
        enable_candidates=True, 
        enable_llm_refine=True,
        llm_enabled=False,  # Отключаем LLM для теста
    )
    
    print("Settings loaded:")
    print(f"  enable_yamnet: {settings.enable_yamnet}")
    print(f"  enable_candidates: {settings.enable_candidates}")
    print(f"  enable_llm_refine: {settings.enable_llm_refine}")
    print(f"  yamnet_threshold: {settings.yamnet_threshold}")
    print(f"  candidates_max_count: {settings.candidates_max_count}")
    print(f"  llm_refine_max_per_request: {settings.llm_refine_max_per_request}")
    
    # Проверяем что можем импортировать новые хэндлеры
    try:
        from features.fusion.candidate_selection_handler import CandidateSelectionHandler
        from features.nlp.llm_refine_candidates_handler import LLMRefineCandidatesHandler
        print("✓ New handlers imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Проверяем что можем создать хэндлеры
    try:
        candidate_handler = CandidateSelectionHandler(
            max_candidates=settings.candidates_max_count,
            min_duration=settings.candidates_min_duration,
            max_duration=settings.candidates_max_duration,
        )
        print("✓ CandidateSelectionHandler created")
        
        from llm.null_client import NullLLMClient
        null_client = NullLLMClient()
        
        refine_handler = LLMRefineCandidatesHandler(
            llm_client=null_client,
            max_candidates_per_request=settings.llm_refine_max_per_request,
        )
        print("✓ LLMRefineCandidatesHandler created")
        
    except Exception as e:
        print(f"✗ Handler creation failed: {e}")
        return False
    
    # Тест candidate selection логики
    test_context = {
        "highlights": [
            {"start": 10.0, "end": 25.0, "score": 0.8, "type": "action"},
            {"start": 50.0, "end": 65.0, "score": 0.7, "type": "comedy"}
        ],
        "scenes": [
            {"start": 0.0, "end": 30.0},
            {"start": 30.0, "end": 60.0},
            {"start": 60.0, "end": 90.0}
        ],
        "timeline": [
            {"time": 15.0, "interest": 0.9},
            {"time": 16.0, "interest": 0.8},
            {"time": 55.0, "interest": 0.7}
        ],
        "laughter_timeline": [
            {"time": 20.0, "prob": 0.8},
            {"time": 21.0, "prob": 0.9},
            {"time": 22.0, "prob": 0.7}
        ],
        "audio_events": [
            {"time": 45.0, "type": "loud", "intensity": 0.8}
        ]
    }
    
    result_context = candidate_handler.handle(test_context.copy())
    candidates = result_context.get("candidates", [])
    
    print(f"✓ Generated {len(candidates)} candidates")
    for i, c in enumerate(candidates):
        print(f"  Candidate {i+1}: {c['start']:.1f}s-{c['end']:.1f}s, score={c['score']:.2f}, reasons={c['reasons']}")
    
    # Тест LLM refine с null client
    if candidates:
        # Добавляем текст для кандидатов
        for c in candidates:
            c["text"] = f"Sample text for segment {c['start']:.1f}s-{c['end']:.1f}s"
        
        refine_context = {"candidates": candidates, "transcript_segments": []}
        refined_context = refine_handler.handle(refine_context)
        refined = refined_context.get("refined_candidates", [])
        
        print(f"✓ Refined {len(refined)} candidates")
        for i, r in enumerate(refined):
            print(f"  Refined {i+1}: {r['title']}, confidence={r['confidence']:.2f}")
    
    return True


def test_context_compatibility():
    """Тест обратной совместимости context ключей."""
    
    # Симулируем старый context
    old_context = {
        "laughter_timeline": [],
        "laughter_summary": {"max_prob": 0.0, "mean_prob": 0.0, "count_positive": 0, "threshold": 0.6},
        "candidates": [],
        "refined_candidates": []
    }
    
    # Проверяем что новые ключи не ломают JSON сериализацию
    try:
        json_str = json.dumps(old_context, indent=2)
        parsed = json.loads(json_str)
        print("✓ New context keys are JSON serializable")
        print("Sample context keys:", list(parsed.keys()))
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing new pipeline features...\n")
    
    success = True
    success &= test_pipeline_with_new_features()
    print()
    success &= test_context_compatibility()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)