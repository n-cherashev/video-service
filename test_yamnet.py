#!/usr/bin/env python3
"""
Простой тест для проверки YAMNet интеграции.
"""

import numpy as np
from ml.yamnet_client import YAMNetClient


def test_yamnet_aggregation():
    """Тест агрегации фреймов YAMNet в секунды."""
    
    # Симуляция данных
    frame_prob = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3])  # 6 фреймов
    duration_seconds = 3.0
    
    client = YAMNetClient()
    timeline = client._to_1s_timeline(frame_prob, duration_seconds)
    
    print("Frame probabilities:", frame_prob)
    print("Timeline (1s):", timeline)
    
    # Проверяем что получили 3 секунды
    assert len(timeline) == 3
    
    # Проверяем что взяли максимум по фреймам в каждой секунде
    # Фреймы 0,1 -> секунда 0: max(0.1, 0.8) = 0.8
    # Фреймы 2,3 -> секунда 1: max(0.9, 0.2) = 0.9  
    # Фреймы 4,5 -> секунда 2: max(0.7, 0.3) = 0.7
    
    expected_probs = [0.8, 0.9, 0.7]
    actual_probs = [item["prob"] for item in timeline]
    
    print("Expected:", expected_probs)
    print("Actual:", actual_probs)
    
    for i, (expected, actual) in enumerate(zip(expected_probs, actual_probs)):
        assert abs(actual - expected) < 0.001, f"Second {i}: expected {expected}, got {actual}"
    
    print("✓ YAMNet aggregation test passed")


def test_class_matching():
    """Тест матчинга классов по подстрокам."""
    
    class_names = [
        "Speech",
        "Male speech, man speaking", 
        "Laughter",
        "Applause",
        "Cheer",
        "Music",
        "Silence"
    ]
    
    client = YAMNetClient()
    client._class_names = class_names
    indices = client._get_target_indices()
    
    print("Class names:", class_names)
    print("Target indices:", indices)
    print("Matched classes:", [class_names[i] for i in indices])
    
    # Должны найти индексы для Laughter (2), Applause (3), Cheer (4)
    expected_indices = [2, 3, 4]
    assert indices == expected_indices, f"Expected {expected_indices}, got {indices}"
    
    print("✓ Class matching test passed")


if __name__ == "__main__":
    test_yamnet_aggregation()
    test_class_matching()
    print("\n✅ All tests passed!")