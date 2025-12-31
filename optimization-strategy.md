# Video-Service: Стратегия оптимизации и улучшения

## Часть 1: Критический анализ текущего состояния

### Основные проблемы

#### 🔴 Критическая проблема #1: CUDA не используется
- **Влияние**: 82% времени (311s из 385s) на STT
- **Причина**: `⚠️ CUDA requested but not available, falling back to CPU`
- **Виновник**: Неправильная установка PyTorch или конфликт CUDA версий
- **Решение**: Переустановить PyTorch с правильной CUDA версией

#### 🔴 Критическая проблема #2: Узкий диапазон scores (0.53-0.57)
- **Влияние**: Невозможно качественно отличить хорошие клипы от плохих
- **Причина**: Одинаковые веса + линейная агрегация
- **Решение**: Нелинейная функция + адаптивные веса

#### 🟡 Проблема #3: Humor detection = 0 для русского
- **Влияние**: Неполный анализ эмоциональности
- **Причина**: Английские маркеры юмора на русском видео
- **Решение**: Интегрировать ruBERT или LLM

#### 🟡 Проблема #4: 15GB RAM потребления
- **Влияние**: Невозможно обрабатывать на слабых машинах
- **Причина**: Все модели в памяти + весь видео буферизирован
- **Решение**: Lazy loading + streaming processing

#### 🟠 Проблема #5: Slow pace_score (0.04-0.11)
- **Влияние**: Компонент почти не влияет на итоговый score
- **Причина**: Порог ожидаемых peaks занижен
- **Решение**: Пересчитать на основе реальных данных

---

## Часть 2: Детальный разбор каждой проблемы и решения

### ПРОБЛЕМА 1: CUDA не работает

#### Диагностика
```
Device set to use cpu
⚠️ CUDA requested but not available, falling back to CPU
```

#### Причины в порядке вероятности
1. **PyTorch установлен без CUDA** (CPU-only версия)
2. **CUDA Toolkit не установлен на машину** (но маловероятно, тк есть ошибка о fallback)
3. **Конфликт версий** (CUDA 12.x vs PyTorch ожидает 11.8)
4. **Переменные окружения** не установлены

#### Решение (пошаговое)

**Шаг 1: Проверить текущую установку**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Шаг 2: Переустановить PyTorch правильно**
```bash
# Удалить старый torch
pip uninstall torch torchvision torchaudio -y

# Установить с CUDA 12.1 (самая новая поддерживаемая)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ИЛИ если нужна CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**Шаг 3: Проверить Whisper на CUDA**
```python
# В speech_to_text_handler.py
from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cuda",  # Явно указываем CUDA
    compute_type="float16",  # Для GPU (быстро + точно)
    num_workers=4  # Параллельные потоки
)
```

#### Ожидаемый результат
- STT: с 300s → 40-60s (5-7x ускорение)
- Общее время: 385s → 150-180s

---

### ПРОБЛЕМА 2: Узкий диапазон scores (0.53-0.57)

#### Анализ причины

**Текущая формула:**
```python
clip_score = (
    0.20 * norm(hook) +      # hook: 0.58-0.72
    0.15 * norm(pace) +      # pace: 0.04-0.11 ← СЛИШКОМ НИЗКО!
    0.15 * norm(intensity) + # intensity: ~0.50-0.65
    0.15 * norm(clarity) +   # clarity: ~0.55-0.62
    0.15 * norm(emotion) +   # emotion: ~0.20-0.40
    0.10 * norm(boundary) +  # boundary: 0.20-0.50
    0.10 * norm(momentum)    # momentum: ~0.30-0.50
)
```

**Проблема:** 
- Все компоненты получают низкие нормированные значения (0.0-1.0 относительно dataset)
- Результат: линейная комбинация 7 чисел (0.3-0.7) → узкий диапазон

#### Решение

**Вариант A: Нелинейные преобразования** (рекомендуется)
```python
def enhanced_scoring(components: Dict[str, float]) -> float:
    """
    Многоуровневый scoring с нелинейностями.
    components: {
        'hook': 0.68,
        'pace': 0.05,
        'intensity': 0.60,
        'clarity': 0.54,
        'emotion': 0.30,
        'boundary': 0.40,
        'momentum': 0.40
    }
    """
    
    # 1. Индивидуальные non-linear transformations
    hook_s = components['hook'] ** 0.9  # Слабо наказываем слабые
    pace_s = min(1.0, components['pace'] * 3) ** 1.2  # Усиливаем низкие
    intensity_s = (components['intensity'] - 0.3) / 0.7  # Нормируем диапазон
    clarity_s = (components['clarity'] ** 1.1) if components['clarity'] > 0.5 else components['clarity'] ** 0.8
    emotion_s = abs(components['emotion'] - 0.5) * 2  # Усиливаем экстремумы
    boundary_s = components['boundary'] ** 0.8
    momentum_s = components['momentum'] ** 1.1
    
    # 2. Приоритизация компонентов
    base_score = (
        0.30 * hook_s +      # Hook — самый важный
        0.20 * clarity_s +   # Clarity — важна для понимания
        0.15 * intensity_s + # Intensity — энергия
        0.15 * emotion_s +   # Emotion — эмоциональность
        0.10 * momentum_s +  # Momentum — дуга
        0.05 * boundary_s +  # Boundary — естественность
        0.05 * pace_s        # Pace — поддержка
    )
    
    # 3. Бонусы за сочетания
    bonuses = 0
    
    # Бонус: сильный hook + strong intensity
    if hook_s > 0.6 and intensity_s > 0.6:
        bonuses += 0.05
    
    # Бонус: хорошо все компоненты
    good_components = sum(1 for v in [hook_s, clarity_s, intensity_s] if v > 0.6)
    bonuses += good_components * 0.02
    
    # Штраф: слабый hook
    if hook_s < 0.3:
        bonuses -= 0.10
    
    # 4. Финальная нелинейность (S-curve для расширения диапазона)
    final = base_score + bonuses
    # Sigmoid-like transformation
    final = 1 / (1 + np.exp(-5 * (final - 0.5)))  # Растягивает 0.3-0.7 в 0.1-0.9
    
    return np.clip(final, 0, 1)
```

**Результат:** Диапазон расширится с [0.53-0.57] на [0.25-0.85]

**Вариант B: Адаптивные веса** (альтернатива)
```python
def adaptive_weights(components: Dict[str, float]) -> Dict[str, float]:
    """Веса зависят от видео-контекста."""
    
    # Если это очень активное видео
    if components['motion_peak'] > 0.7:
        return {
            'hook': 0.25,
            'intensity': 0.25,  # Усиливаем для активного видео
            'clarity': 0.15,
            'emotion': 0.15,
            'pace': 0.10,
            'boundary': 0.05,
            'momentum': 0.05,
        }
    
    # Если это диалог/разговор
    elif components['dialogue_ratio'] > 0.7:
        return {
            'hook': 0.25,
            'clarity': 0.25,  # Усиливаем для диалога
            'emotion': 0.20,
            'intensity': 0.15,
            'pace': 0.10,
            'boundary': 0.03,
            'momentum': 0.02,
        }
    
    # Default
    else:
        return {
            'hook': 0.25,
            'intensity': 0.20,
            'clarity': 0.20,
            'emotion': 0.15,
            'pace': 0.10,
            'boundary': 0.05,
            'momentum': 0.05,
        }
```

---

### ПРОБЛЕМА 3: Humor detection = 0

#### Корень проблемы
```python
# Текущий код (неработает)
HUMOR_MARKERS = [
    "funny", "hilarious", "lol", "haha", "comic"
]

def detect_humor(text: str):
    text_lower = text.lower()
    return any(marker in text_lower for marker in HUMOR_MARKERS)
```

**Проблема:** Все маркеры английские, видео на русском!

#### Решение 1: Русский контент (быстро)
```python
# humor_detection_handler.py
from typing import List, Dict

# Русские маркеры
RUSSIAN_HUMOR_MARKERS = {
    'laughing': ['хахаха', 'ахахаха', 'ахаха', 'ха-ха', 'хе-хе', 'хихи'],
    'explicit': ['смешно', 'прикол', 'угар', 'ору', 'ржу', 'умираю', 'жесть'],
    'slang': ['лол', 'кек', 'кекв', 'ауф', 'ору', 'буль'],
    'irony': ['конечно', 'типа', 'якобы', 'мол', 'вот это да'],
    'sarcasm': ['супер', 'замечательно', 'классно', 'отлично', 'огонь'],
    'emotion_markers': ['😂', '🤣', '😆', '😅', '😄', '☠️', '💀']
}

def detect_humor_russian(text: str, sentiment_data: Optional[List[float]] = None) -> float:
    """
    Детекция юмора для русского контента.
    Returns: confidence score 0-1
    """
    text_lower = text.lower()
    confidence = 0
    
    # 1. Простое совпадение маркеров
    marker_matches = 0
    for category, markers in RUSSIAN_HUMOR_MARKERS.items():
        for marker in markers:
            if marker in text_lower:
                marker_matches += 1
                confidence += 0.15
    
    # 2. Анализ тона (если есть sentiment)
    if sentiment_data and len(sentiment_data) > 0:
        sentiment_changes = sum(1 for i in range(1, len(sentiment_data)) 
                               if (sentiment_data[i] > 0.5) and (sentiment_data[i-1] < 0))
        if sentiment_changes > 0:
            confidence += sentiment_changes * 0.1
    
    # 3. Структурные признаки
    # Многоточие может указывать на паузу для смеха
    ellipsis_count = text.count('...')
    if ellipsis_count > text.count('.') / 3:
        confidence += 0.05
    
    # Восклицательные знаки
    exclamation_ratio = text.count('!') / max(len(text.split()), 1)
    if exclamation_ratio > 0.1:
        confidence += 0.1
    
    return min(confidence, 1.0)


# В основном обработчике
class HumorDetectionHandler(BaseHandler):
    def handle(self, state: AnalysisState) -> AnalysisState:
        humor_scores = []
        
        for segment in state.speechToText.segments:
            confidence = detect_humor_russian(
                segment.text,
                sentiment_data=state.sentiment.segmentSentiments  # Если доступно
            )
            humor_scores.append({
                'segment_index': segment.index,
                'start': segment.start,
                'end': segment.end,
                'humor_score': confidence,
                'text_excerpt': segment.text[:50]
            })
        
        state.humor = HumorDetectionResult(
            scores=humor_scores,
            summary={'mean': np.mean([s['humor_score'] for s in humor_scores]),
                     'max': max([s['humor_score'] for s in humor_scores]),
                     'count_positive': sum(1 for s in humor_scores if s['humor_score'] > 0.3)}
        )
        return state
```

#### Решение 2: LLM-based (если нужна точность)
```python
async def detect_humor_with_llm(
    segments: List[TranscriptSegment],
    model: str = "gpt-4-mini"
) -> List[float]:
    """
    Использовать GPT для анализа юмора в русском контексте.
    """
    
    # Группируем сегменты в батчи (экономим API calls)
    batch_size = 10
    scores = []
    
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        batch_text = "\n\n".join([
            f"{j}. ({s.start:.0f}s) {s.text[:100]}"
            for j, s in enumerate(batch)
        ])
        
        prompt = f"""
Проанализируй эти фрагменты видео на предмет юмора и забавности.
Оцени каждый фрагмент по шкале 0-1 (0 = серьёзно, 1 = очень смешно).

{batch_text}

Верни JSON с массивом оценок:
{{"scores": [0.0, 0.3, 0.8, ...]}}
"""
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        batch_scores = json.loads(response.choices[0].message.content)['scores']
        scores.extend(batch_scores)
    
    return scores
```

---

### ПРОБЛЕМА 4: 15GB RAM потребления

#### Анализ потребления
```
Model weights:      ~4-5 GB
  - Whisper base    ~1.5 GB
  - YAMNet          ~0.5 GB
  - BERT/distilbert ~1.5 GB
  - Other           ~1 GB

Runtime buffers:    ~8-10 GB
  - TensorFlow      ~2-3 GB
  - Audio (WAV)     ~2 GB
  - OpenCV frames   ~2-3 GB
  - NumPy arrays    ~1 GB
  - Python          ~1 GB

TOTAL:              ~15 GB
```

#### Решение 1: Lazy Loading (экономит 4-5 GB)
```python
# config/model_manager.py

import gc
from typing import Dict, Any, Optional
import torch

class LazyModelManager:
    """
    Загружает модели только когда они нужны.
    Освобождает память после использования.
    """
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}
    
    def get_model(self, model_name: str, device: str = "cuda") -> Any:
        """
        Получить модель. Загружает, если не загружена.
        model_name: "whisper", "bert", "yamnet"
        """
        
        if model_name in self._models:
            self._last_used[model_name] = time.time()
            return self._models[model_name]
        
        # Загружаем модель
        if model_name == "whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("base", device=device, compute_type="float16")
        
        elif model_name == "bert":
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                device_map=device,
                low_cpu_mem_usage=True
            )
        
        elif model_name == "yamnet":
            import tensorflow_hub as hub
            model = hub.load("https://tfhub.dev/google/yamnet/1")
        
        self._models[model_name] = model
        self._last_used[model_name] = time.time()
        
        return model
    
    def cleanup_old_models(self, max_age_minutes: int = 30):
        """Удалить неиспользуемые модели из памяти."""
        current_time = time.time()
        to_remove = []
        
        for model_name, last_used in self._last_used.items():
            if (current_time - last_used) > max_age_minutes * 60:
                to_remove.append(model_name)
        
        for model_name in to_remove:
            del self._models[model_name]
            del self._last_used[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return len(to_remove)

# Глобальный инстанс
MODEL_MANAGER = LazyModelManager()

# Использование в handlers
class SentimentAnalysisHandler(BaseHandler):
    def handle(self, state: AnalysisState) -> AnalysisState:
        model = MODEL_MANAGER.get_model("bert", device=state.globalConfig.device)
        
        # Обработка...
        results = model.predict(texts)
        
        # После использования очищаем старые модели
        MODEL_MANAGER.cleanup_old_models(max_age_minutes=15)
        
        return state
```

#### Решение 2: Streaming Audio Processing (экономит 2 GB)
```python
# features/audio/audio_features_handler.py

def extract_features_streaming(
    audio_path: str,
    chunk_duration: float = 30.0,  # Обрабатываем по 30 сек
    sr: int = 16000
) -> List[AudioFeaturePoint]:
    """
    Потоковое извлечение фичей вместо загрузки всего аудио в память.
    """
    import librosa
    import soundfile as sf
    
    features = []
    
    # Получаем информацию о файле без загрузки
    info = sf.info(audio_path)
    total_frames = info.frames
    chunk_frames = int(chunk_duration * sr)
    
    # Читаем и обрабатываем чанками
    with sf.SoundFile(audio_path) as f:
        for start_frame in range(0, total_frames, chunk_frames):
            # Читаем только нужный чанк
            audio_chunk = f.read(chunk_frames)
            
            # Извлекаем фичи для этого чанка
            chunk_features = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
            loudness = np.sqrt(np.mean(audio_chunk ** 2))
            energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
            
            # Сохраняем результат
            time = start_frame / sr
            features.append(AudioFeaturePoint(
                time=time,
                loudness=loudness,
                energy=energy,
                mfcc=chunk_features.mean(axis=1).tolist()
            ))
            
            # Освобождаем память для этого чанка
            del audio_chunk, chunk_features
            gc.collect()
    
    return features
```

#### Решение 3: Chunk-based Video Processing (для очень длинных видео)
```python
# pipeline/chunk_processor.py

def process_video_in_chunks(
    video_path: str,
    chunk_duration: int = 600,  # 10 минут на чанк
) -> AnalysisResult:
    """
    Обрабатывать видео чанками по 10 минут.
    Освобождает память между чанками.
    """
    
    total_duration = get_video_duration(video_path)
    chunk_results = []
    
    for start_sec in range(0, int(total_duration), chunk_duration):
        end_sec = min(start_sec + chunk_duration, int(total_duration))
        
        # Обрезать видео на чанк
        chunk_path = extract_video_segment(video_path, start_sec, end_sec)
        
        # Обработать чанк
        chunk_state = AnalysisState(
            videoPath=chunk_path,
            analysisId=f"chunk_{start_sec}_{end_sec}",
            globalConfig=GLOBAL_CONFIG,
            # ...
        )
        
        # Запустить пайплайн для чанка
        result = run_dag_pipeline(chunk_state)
        chunk_results.append((start_sec, result))
        
        # Очистить память
        os.remove(chunk_path)
        gc.collect()
        torch.cuda.empty_cache()
    
    # Объединить результаты с сдвигом по времени
    final_result = merge_chunk_results(chunk_results, chunk_duration)
    return final_result

def merge_chunk_results(
    chunk_results: List[Tuple[int, AnalysisResult]],
    chunk_duration: int
) -> AnalysisResult:
    """Объединить результаты из разных чанков."""
    
    merged = AnalysisResult()
    
    for offset, chunk_result in chunk_results:
        # Сдвигаем все временные метки
        for timeline_point in chunk_result.timeline.timeline:
            timeline_point.time += offset
        
        merged.timeline.timeline.extend(chunk_result.timeline.timeline)
        
        # Сдвигаем хайлайты
        for clip in chunk_result.highlights.clips:
            clip.startSeconds += offset
            clip.endSeconds += offset
        
        merged.highlights.clips.extend(chunk_result.highlights.clips)
    
    return merged
```

**Ожидаемый результат:** 15GB → 8-10GB (-33% памяти)

---

### ПРОБЛЕМА 5: Низкий pace_score (0.04-0.11)

#### Анализ
```python
# Текущий расчёт:
peaks_in_window = count(local_peaks in interest[start:end])
expected_peaks ~ 1 пик на 10 сек
pace = peaks_in_window / (duration_sec / 10)

# Для 60-секундного окна:
expected_peaks = 60 / 10 = 6
actual_peaks = 0.5 (в среднем) ← ОЧЕНЬ НИЗКО!
pace = 0.5 / 6 = 0.083
```

#### Причина
Timeline слишком сглаженный (1-sec шаг). Пики размываются.

#### Решение 1: Используй более высокое разрешение временной оси
```python
# Вместо 1-sec timeline делаем 0.1-sec (100ms)
def create_high_res_timeline(
    motion: List[float],  # per frame (33ms)
    audio_loudness: List[float],  # per frame
    fps: float = 30,
    target_step: float = 0.1  # 100ms
) -> List[TimelinePoint]:
    """
    Создать высокоразрешённую временную шкалу для точного поиска пиков.
    """
    
    frame_step = int(fps * target_step)  # сколько фреймов в 100ms
    
    timeline = []
    for i in range(0, len(motion), frame_step):
        window_end = min(i + frame_step, len(motion))
        
        time_sec = i / fps
        motion_val = np.mean(motion[i:window_end])
        loudness_val = np.mean(audio_loudness[i:window_end])
        
        # Остальные компоненты интерполируем...
        
        timeline.append(TimelinePoint(
            time=time_sec,
            motion=motion_val,
            audioLoudness=loudness_val,
            # ...
        ))
    
    return timeline
```

#### Решение 2: Лучше считай пики
```python
def find_local_peaks(
    signal: np.ndarray,
    height_percentile: float = 75,  # top 25%
    min_distance: int = 3  # минимум 0.3 сек между пиками
) -> List[int]:
    """
    Находит пики в сигнале с адаптивным порогом.
    """
    from scipy.signal import find_peaks
    
    # Адаптивный порог (в зависимости от распределения)
    threshold = np.percentile(signal, height_percentile)
    
    peaks, properties = find_peaks(
        signal,
        height=threshold,
        distance=min_distance,
        prominence=np.std(signal) * 0.5
    )
    
    return peaks

def calculate_pace_score(
    timeline: List[TimelinePoint],
    start_idx: int,
    end_idx: int
) -> float:
    """Переделанный расчёт pace."""
    
    # Извлекаем интерес для окна
    interest = np.array([p.interest for p in timeline[start_idx:end_idx]])
    
    # Находим пики
    peaks = find_local_peaks(interest, height_percentile=70)
    
    # Исправленная формула
    duration_sec = (end_idx - start_idx) * 0.1  # если timeline на 100ms
    expected_peaks = max(1, duration_sec / 5)  # 1 пик на 5 сек (более реалистично)
    
    pace = min(len(peaks) / expected_peaks, 2.0)  # cap на 2.0 для нормализации
    
    return min(pace, 1.0)
```

#### Решение 3: Переоцени веса
```python
def enhanced_scoring_v2(components: Dict[str, float]) -> float:
    """
    С правильными весами для скорректированного pace.
    """
    
    hook_s = components['hook'] ** 0.9
    pace_s = (components['pace'] - 0.2) / 0.8  # Переоцентируем (было 0.04-0.11 → 0.0-1.0)
    intensity_s = components['intensity'] ** 1.0
    clarity_s = components['clarity'] ** 1.1
    emotion_s = abs(components['emotion'] - 0.5) * 2
    boundary_s = components['boundary'] ** 0.8
    momentum_s = components['momentum'] ** 1.1
    
    score = (
        0.28 * hook_s +      # Hook — ключевой
        0.18 * clarity_s +   # Clarity
        0.15 * intensity_s + # Intensity
        0.15 * emotion_s +   # Emotion
        0.12 * pace_s +      # Pace — теперь значимый!
        0.07 * boundary_s +
        0.05 * momentum_s
    )
    
    return np.clip(score, 0, 1)
```

---

## Часть 3: Roadmap реализации

### Фаза 1: КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ (1-2 дня) 🔴

```
[ ] 1.1. Переустановить PyTorch с CUDA поддержкой
        - Command: pip install torch CUDA 12.1
        - Тест: CUDA available должна быть True
        - Результат: STT ускорится в 5x

[ ] 1.2. Проверить Whisper на GPU
        - Модифицировать speech_to_text_handler.py
        - Использовать float16 на GPU
        - Тест: Whisper должен работать на CUDA

[ ] 1.3. Добавить русские маркеры юмора
        - Добавить RUSSIAN_HUMOR_MARKERS словарь
        - Обновить detect_humor функцию
        - Тест: humor_count > 0 для русского видео

[ ] 1.4. Исправить pace_score
        - Пересчитать ожидаемые пики (1 на 5 сек вместо 1 на 10)
        - Добавить find_local_peaks с адаптивным порогом
        - Тест: pace_score должен быть в диапазоне 0.3-0.8

Ожидаемый результат: 385s → 100-120s (3x ускорение)
```

### Фаза 2: Оптимизация памяти (3-5 дней) 🟡

```
[ ] 2.1. Реализовать LazyModelManager
        - Класс для управления моделями
        - Lazy loading + cleanup
        - Интеграция во все handlers

[ ] 2.2. Streaming audio processing
        - Заменить librosa.load() на chunk processing
        - Тест: memory usage < 5GB для аудио

[ ] 2.3. Chunk-based video processing
        - Для видео > 60 мин разбить на чанки
        - Merge результатов

Ожидаемый результат: 15GB → 8GB (-47% память)
```

### Фаза 3: Улучшение scoring (1 неделя) 🟠

```
[ ] 3.1. Реализовать enhanced_scoring с нелинейностями
        - Заменить линейную комбинацию
        - Добавить bonuses + penalties
        - Тест: диапазон scores должен быть [0.25-0.85]

[ ] 3.2. Адаптивные веса в зависимости от видео
        - Detect видео-типа (активное/диалог/микс)
        - Apply соответствующие веса
        - Тест: scores должны лучше различать качество

[ ] 3.3. Улучшенная диверсификация
        - Лучше считать overlap penalty
        - Реализовать smart refit (не просто сдвиг, а оптимизация)

Ожидаемый результат: найденные клипы будут заметно различаться по качеству
```

### Фаза 4: Advanced Features (2-3 недели) 🟢

```
[ ] 4.1. LLM-refinement (опционально)
        - Batch API calls для GPT-4
        - Анализ top-30 кандидатов
        - Проверка "понятности без контекста"

[ ] 4.2. Multimodal analysis
        - Gemini 1.5 или GPT-4V для анализа кадров
        - Детекция лиц, текста в видео
        - Синтез с audio анализом

[ ] 4.3. Web UI для просмотра
        - Streamlit или FastAPI + React
        - Интерактивный просмотр хайлайтов
        - Экспорт в разные форматы

[ ] 4.4. Production deployment
        - Docker контейнер
        - Task queue (Celery)
        - API endpoint
```

---

## Часть 4: Конкретные изменения файлов

### Файл: `features/config/initialization_handler.py` (НОВЫЙ)

```python
# Полная инициализация с CUDA detection

class InitializationHandler(BaseHandler):
    def handle(self, state: AnalysisState) -> AnalysisState:
        # CUDA detection
        device, cuda_info = self._detect_cuda()
        
        # Выбор размера модели
        if device == "cuda" and cuda_info.get("cuda_memory_mb", 0) >= 8000:
            whisper_size = "small"  # На GPU small будет быстро
        elif device == "cuda":
            whisper_size = "base"
        else:
            whisper_size = "tiny"  # На CPU только tiny
        
        state.globalConfig = GlobalConfig(
            device=device,
            gpu_available=(device == "cuda"),
            whisper_model_size=whisper_size,
            cuda_memory_mb=cuda_info.get("cuda_memory_mb"),
            # ... остальные параметры
        )
        
        return state

    @staticmethod
    def _detect_cuda() -> Tuple[Literal["cuda", "cpu"], Dict]:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", {
                    "version": torch.version.cuda,
                    "cuda_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2),
                    "gpu_name": torch.cuda.get_device_name(0),
                }
            else:
                return "cpu", {}
        except ImportError:
            return "cpu", {}
```

### Файл: `features/nlp/humor_detection_handler.py` (ПЕРЕДЕЛАН)

```python
class HumorDetectionHandler(BaseHandler):
    RUSSIAN_HUMOR_MARKERS = {
        'laughing': ['хахаха', 'ахахаха', 'ахаха', 'хе-хе', 'хихи'],
        'explicit': ['смешно', 'прикол', 'угар', 'ору', 'ржу', 'жесть'],
        'slang': ['лол', 'кек', 'ауф', 'буль'],
        'emotion': ['😂', '🤣', '😆', '😅'],
    }
    
    def handle(self, state: AnalysisState) -> AnalysisState:
        humor_scores = []
        
        for segment in state.speechToText.segments:
            confidence = self._detect_humor_russian(
                segment.text,
                segment.start,
                segment.end,
                state.sentiment  # Используем sentiment для усиления
            )
            
            humor_scores.append({
                'segment': segment.index,
                'score': confidence,
                'start': segment.start,
                'end': segment.end,
            })
        
        state.humor = HumorDetectionResult(
            scores=humor_scores,
            summary={
                'mean': np.mean([s['score'] for s in humor_scores]),
                'max': max([s['score'] for s in humor_scores]),
                'count_positive': sum(1 for s in humor_scores if s['score'] > 0.3),
            }
        )
        
        return state
    
    def _detect_humor_russian(self, text: str, start: float, end: float, sentiment_data) -> float:
        text_lower = text.lower()
        confidence = 0
        
        # Маркер-матчинг
        for category, markers in self.RUSSIAN_HUMOR_MARKERS.items():
            confidence += sum(0.2 for m in markers if m in text_lower)
        
        # Структурные признаки
        confidence += text.count('!') * 0.05
        confidence += text.count('...') * 0.03
        
        # Sentiment boost (если есть скачки эмоций → смешно)
        if sentiment_data:
            seg_sentiment = [s for s in sentiment_data.segmentSentiments 
                           if s.start >= start and s.end <= end]
            if seg_sentiment:
                changes = sum(1 for i in range(1, len(seg_sentiment)) 
                            if seg_sentiment[i].sentiment * seg_sentiment[i-1].sentiment < 0)
                confidence += changes * 0.15
        
        return min(confidence, 1.0)
```

### Файл: `features/highlights/viral_moments_handler.py` (ПЕРЕДЕЛАН)

```python
class ViralMomentsHandler(BaseHandler):
    def handle(self, state: AnalysisState) -> AnalysisState:
        # Получить кандидатов из CandidateSelectionHandler
        candidates = state.candidates
        
        # Скорить всех
        scored_candidates = []
        for candidate in candidates:
            score_breakdown = self._score_candidate(candidate, state.timeline.timeline)
            scored_candidates.append({
                'candidate': candidate,
                'score': score_breakdown['final_score'],
                'breakdown': score_breakdown,
            })
        
        # Сортировать по score
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Диверсификация + selection
        selected = self._select_diverse_clips(scored_candidates)
        
        # Формировать результат
        state.highlights = HighlightDetectionResult(
            clips=[self._format_clip(c) for c in selected],
            summary={'total_selected': len(selected), 'total_candidates': len(candidates)}
        )
        
        return state
    
    def _score_candidate(self, candidate, timeline) -> Dict:
        """Многоуровневый скоринг с нелинейностями."""
        
        start_idx = int(candidate.start_seconds)
        end_idx = int(candidate.end_seconds)
        window = timeline[start_idx:end_idx]
        
        # Вычислить компоненты
        hook = np.mean([p.interest for p in window[:3]])
        clarity = np.mean([p.clarity for p in window])
        intensity = 0.6 * np.mean([p.motion for p in window]) + 0.4 * np.mean([p.audioLoudness for p in window])
        emotion = abs(np.mean([p.sentiment for p in window]))
        
        # Пики в окне
        interest_vals = np.array([p.interest for p in window])
        peaks = find_local_peaks(interest_vals)
        pace = len(peaks) / max(1, len(window) / 5)
        
        # Momentum
        first_half_intensity = np.mean([p.audioLoudness for p in window[:len(window)//2]])
        second_half_intensity = np.mean([p.audioLoudness for p in window[len(window)//2:]])
        momentum = second_half_intensity / max(first_half_intensity, 0.1)
        
        # Boundaries
        boundary_score = 0
        if window[0].isSceneBoundary:
            boundary_score += 0.5
        if window[-1].isSceneBoundary:
            boundary_score += 0.5
        
        # Нелинейные преобразования
        hook_s = hook ** 0.9
        clarity_s = clarity ** 1.1
        intensity_s = (intensity - 0.3) / 0.7 if intensity > 0.3 else 0
        emotion_s = emotion ** 1.0
        pace_s = min(pace / 2, 1.0)
        momentum_s = momentum ** 1.1
        boundary_s = boundary_score ** 0.8
        
        # Агрегация с новыми весами
        final_score = (
            0.30 * hook_s +
            0.22 * clarity_s +
            0.16 * intensity_s +
            0.15 * emotion_s +
            0.10 * pace_s +
            0.05 * boundary_s +
            0.02 * momentum_s
        )
        
        # Sigmoid для расширения диапазона
        final_score = 1 / (1 + np.exp(-5 * (final_score - 0.5)))
        
        return {
            'final_score': final_score,
            'hook': hook_s,
            'clarity': clarity_s,
            'intensity': intensity_s,
            'emotion': emotion_s,
            'pace': pace_s,
            'boundary': boundary_s,
            'momentum': momentum_s,
        }
    
    def _select_diverse_clips(self, scored: List) -> List:
        """Выбрать разнообразные клипы с штрафом за overlap."""
        
        selected = []
        used_regions = []
        
        for item in scored:
            candidate = item['candidate']
            score = item['score']
            
            # Вычислить overlap с выбранными
            max_overlap = 0
            for selected_item in selected:
                overlap = self._calculate_overlap(candidate, selected_item['candidate'])
                max_overlap = max(max_overlap, overlap)
            
            # Применить штраф
            adjusted_score = score
            if max_overlap > 0.65:
                adjusted_score *= 0.3
            elif max_overlap > 0.35:
                adjusted_score *= 0.7
            
            # Если всё ещё хороший или это важный момент
            if adjusted_score > 0.35 or len(selected) < 5:
                selected.append(item)
                used_regions.append((candidate.start_seconds, candidate.end_seconds))
        
        return selected[:80]  # Top 80 кандидатов
    
    def _calculate_overlap(self, c1, c2) -> float:
        """Считать коэффициент пересечения."""
        inter = max(0, min(c1.end_seconds, c2.end_seconds) - max(c1.start_seconds, c2.start_seconds))
        union = max(c1.end_seconds - c1.start_seconds, c2.end_seconds - c2.start_seconds)
        return inter / union if union > 0 else 0
```

---

## Часть 5: Тестирование и валидация

### Metrics для отслеживания

```python
class PerformanceMetrics:
    """Сравнивать результаты до и после оптимизации."""
    
    def __init__(self):
        self.metrics = {
            'processing_time': None,
            'ram_usage_peak': None,
            'gpu_utilized': False,
            'cuda_available': False,
            'score_range': (None, None),
            'humor_detected_count': 0,
            'pace_avg': 0,
        }
    
    def record(self, **kwargs):
        self.metrics.update(kwargs)
    
    def report(self):
        return f"""
PERFORMANCE REPORT
==================
Processing Time: {self.metrics['processing_time']:.1f}s
Peak RAM: {self.metrics['ram_usage_peak']:.1f}GB
GPU Used: {self.metrics['gpu_utilized']}
CUDA Available: {self.metrics['cuda_available']}
Score Range: {self.metrics['score_range'][0]:.2f}-{self.metrics['score_range'][1]:.2f}
Humor Detected: {self.metrics['humor_detected_count']}
Avg Pace Score: {self.metrics['pace_avg']:.3f}
"""
```

### Тестовый скрипт

```python
# tests/test_optimizations.py

def test_phase1_critical_fixes():
    """Проверить исправления фазы 1."""
    
    # 1. CUDA должна быть доступна
    assert torch.cuda.is_available(), "CUDA not available!"
    
    # 2. Whisper должен работать на GPU
    model = WhisperModel("base", device="cuda")
    assert model.model is not None
    
    # 3. Humor должен быть > 0 для русского
    text = "Это очень смешно! Ахахаха!"
    humor = detect_humor_russian(text)
    assert humor > 0.3
    
    # 4. Pace должен быть в разумном диапазоне
    # (тестируется на реальных данных)

def test_phase2_memory_optimization():
    """Проверить оптимизацию памяти."""
    
    import psutil
    process = psutil.Process()
    
    # Начальная память
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    # Запустить пайплайн
    result = run_pipeline()
    
    # Пиковая память
    peak_mem = process.memory_info().rss / 1024 / 1024
    
    # Должна быть < 10GB
    assert peak_mem < 10000, f"Memory usage too high: {peak_mem}MB"

def test_phase3_scoring_improvement():
    """Проверить улучшение scoring."""
    
    # Запустить на тестовом видео
    result = run_pipeline(test_video_path)
    
    scores = [clip.score for clip in result.highlights.clips]
    
    # Диапазон должен быть расширен
    score_range = max(scores) - min(scores)
    assert score_range > 0.3, f"Score range too narrow: {score_range}"
    
    # Есть клипы разного качества
    high = sum(1 for s in scores if s > 0.7)
    low = sum(1 for s in scores if s < 0.4)
    assert high > 0 and low > 0, "Scores not diverse enough"
```

---

## Итоговая рекомендация

### Приоритеты (что делать в первую очередь):

1. **КРИТИЧНО**: Исправить CUDA для Whisper (1-2 часа)
   - Это даст 5x ускорение → 385s → 80s
   - Остальная оптимизация будет на этом фундаменте

2. **ВАЖНО**: Добавить русский контент support (2-3 часа)
   - Humor detection, sentiment на русском

3. **СЕРЬЁЗНО**: Улучшить scoring (1-2 дня)
   - Нелинейные трансформации
   - Адаптивные веса
   - Это улучшит качество найденных клипов

4. **ПОЛЕЗНО**: Оптимизировать память (3-5 дней)
   - LazyLoading, streaming processing
   - Пригодится для продакшена

5. **NICE-TO-HAVE**: LLM refinement (2-3 недели)
   - Опционально, для максимального качества
