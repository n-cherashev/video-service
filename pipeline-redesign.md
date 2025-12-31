# Переосмысленный пайплайн анализа видео для поиска ключевых моментов (Highlights Detection)

## Анализ существующих решений

После анализа интернета выявлены следующие подходы:

### 1. **VideoPipe** (C++, open-source)
- **Архитектура**: Node-based, графовая модель
- **Паттерн**: каждый узел независим, можно комбинировать в DAG
- **Преимущества**: параллелизм, масштабируемость, простота добавления новых узлов
- **Недостатки**: C++, медленная разработка

### 2. **PreenCut** (Python, LLM-centric)
- **Поток**: STT (Whisper) → LLM analysis → segment extraction
- **Упор**: на LLM для интерпретации содержания (не на ML-метрики)
- **Выход**: сегменты с AI-summary и tags
- **Недостаток**: полагается только на LLM, медленно для длинных видео

### 3. **DeepStream** (NVIDIA, C++)
- **Фокус**: real-time video streams, object tracking
- **Архитектура**: pipeline of elements, буферизация между элементами
- **Неподходящ**: для batch-обработки исторических видео

### 4. **VIAME** (Video and Image Analytics)
- **Подход**: модульная, plugin-based система
- **Включает**: детекцию, трекинг, аннотацию
- **Тяжелая**: много зависимостей

---

## Переосмысленный пайплайн: Гибридный подход

**Ключевая идея**: не полагаться только на одну метрику (interest) или только на LLM, а использовать **многоуровневую систему с ранней фильтрацией и поздней валидацией**.

### Архитектурные принципы

1. **Разделение по скорости обработки**:
   - **Быстрый слой** (Fast): motion, audio features, scene detection (параллельно, GPU/CPU)
   - **Средний слой** (Medium): STT, sentiment, quality assessment
   - **Медленный слой** (Slow): LLM refinement (опционально, только для top-N кандидатов)

2. **Ранняя фильтрация**: вместо анализа всех моментов, сначала найти ~100–200 кандидатов (дешево), потом отфильтровать до ~30–50 (дорого).

3. **Графовая архитектура**: каждый обработчик (handler) → узел в DAG, может работать параллельно.

4. **Инкрементальное обогащение состояния**: каждый узел добавляет свои данные к единому `AnalysisState` объекту.

---

## Пересмотренная последовательность хэндлеров

### Слой 1: Инициализация и чтение (Initialization Layer)

#### **1.1. InitializationHandler**
```
Вход: видео-путь, конфиг
Выход: GlobalConfig с CUDA info, analysisId
Побочные эффекты: инициализация логирования, папок temp/output
```

#### **1.2. ReadFileHandler + VideoMetaHandler** (параллельно)
```
ReadFile: проверка доступности, размер файла
VideoMeta: fps, resolution, codec, duration
Параллелизм: оба независимы
```

---

### Слой 2: Быстрые фичи (Fast Feature Extraction) — ВСЁ ПАРАЛЛЕЛЬНО

Все эти четыре хэндлера могут работать одновременно на видеофайле:

#### **2.1. MotionAnalysisHandler** (GPU-friendly)
```
Метод: Frame Diff (быстро) или Optical Flow (точнее, но медленнее)
Выход: motionHeatmap per 1-second
Параллелизм: батчевая обработка кадров (multiprocessing)
GPU: OpenCV CUDA acceleration (если доступна)
Параметры:
  - sample_skip: 3 (каждый 3-й кадр для видео > 10 мин)
  - resize_to: 224 (малое разрешение для скорости)
```

#### **2.2. AudioExtractionHandler + AudioFeaturesHandler** (2 шага)
```
Шаг 1 (AudioExtraction): ffmpeg → WAV, sr=16kHz, mono
Шаг 2 (AudioFeatures): параллельно вычисляем:
  - Loudness (RMS per frame)
  - Energy (спектральная мощность)
  - Speech Probability (VAD, можно silero-vad для легкости)
Выход: timeseriesPoints per ~100ms, потом агрегируем до 1-second
```

#### **2.3. SceneDetectionHandler**
```
Метод: PySceneDetect (AdaptiveDetector) или histogram-based
Выход: список сцен, временные границы
Параллелизм: независим от других
Фильтрация: убираем сцены < 0.5 сек (noise)
```

#### **2.4. AudioEventsHandler**
```
Вход: audioFeatures (loudness)
Действие: детекция пиков (localmaxima) с threshold и фильтрацией
Выход: список событий {start, end, time, type, confidence}
Легко & быстро: просто peak detection
```

---

### Слой 3: Медленные фичи (Medium Feature Extraction)

#### **3.1. SpeechToTextHandler**
```
Модель: OpenAI Whisper (размер зависит от GPU)
Выход: transcriptSegments с временными метками
Важно: caching модели (загружаем один раз)
```

#### **3.2. SentimentAnalysisHandler**
```
Модель: distilbert-sst2 (быстро, <1GB) или multilingual вариант
Вход: transcriptSegments.text
Выход: sentimentTimeline (per 1-second, спроектировано из сегментов)
```

#### **3.3. SpeechQualityHandler** (NEW)
```
Метрики:
  - SNR (Signal-to-Noise Ratio) на основе VAD + loudness
  - Stability: std dev громкости в речевых участках
  - Silence ratio: % молчания
Выход: clarityTimeline (per 1-second)
```

#### **3.4. HumorDetectionHandler** (опционально)
```
Метод: простой (лексико-фонетический анализ) или YAMNet (audio-based)
Выход: humorScores per segment
```

---

### Слой 4: Синтез (Fusion Layer)

#### **4.1. FusionTimelineHandler**
```
Вход: все результаты слоёв 1–3
Действие: объединить в единый 1-second timeline
timeline[i] = {
  time: float,
  motion: float [0-1],
  audioLoudness: float [0-1],
  audioEnergy: float [0-1],
  speechProbability: float [0-1],
  sentiment: float [-1, 1],
  clarity: float [0-1],
  hasLoudSound: bool,
  isSceneBoundary: bool,
  isDialogue: bool,
  interest: float [0-1]  ← агрегированный скор
}

Вычисление interest:
  interest = (
    0.20 * norm(motion) +
    0.18 * norm(loudness) +
    0.15 * norm(energy) +
    0.15 * abs(sentiment) +
    0.12 * clarity +
    0.10 * speechProbability +
    0.10 * (1.0 if hasLoudSound else 0.0)
  )
```

---

### Слой 5: Выбор моментов (Highlights Detection Layer)

#### **5.1. CandidateSelectionHandler** (НОВОЕ)
```
Действие: генерация кандидатов вокруг якорей

ЯКОРЯ (anchors):
1. Пики interest (локальные максимумы, top 20%)
2. Пики motion
3. Пики loudness + audio events
4. Scene boundaries (isSceneBoundary=True)
5. Переходы в диалог (isdialogue: False→True)
6. Резкие смены sentiment (|Δsentiment| > 0.5)

ГЕНЕРАЦИЯ КАНДИДАТОВ:
Для каждого якоря:
  duration_variants = [12, 18, 25, 35, 45, 60] сек
  positioning = [center, hook (первые 2-3 сек), payoff (последние 2-3 сек)]
  → генерируем 3 * 6 = 18 окон вокруг якоря
  
Фильтрация дубликатов: keep unique (start, end) пары

Выход: кандидаты {start, end, duration, anchorType, anchorScore}
```

#### **5.2. HighlightDetectionHandler** (ПЕРЕДЕЛАН)
```
Вход: timeline + candidates из 5.1

СКОРИНГ каждого кандидата:

hook_score = mean(interest[start:min(start+3, end)])
  if hook_score < 0.3: *0.5 (штраф за слабый старт)

pace_score = count(local_peaks in interest[start:end]) / expected_peaks
  expected_peaks ~ 1 пик на 10 секунд

intensity_score = 0.6*mean(motion[start:end]) + 0.4*mean(loudness[start:end])

clarity_score = mean(clarity[start:end])
  if mean(isdialogue) < 0.3: *0.8 (штраф за мало диалога)

emotion_score = (
  max(sentiment) - min(sentiment) / 2.0 +
  count(sign_changes_in_sentiment) * 0.1
)

boundary_score = (
  (1.0 if start near scenebound else 0.0) +
  (1.0 if end near scenebound else 0.0)
) * 0.5

momentum_score = (
  1.0 if end_intensity > start_intensity*1.2 else end_intensity/max(start_intensity, 0.1)
)

ИТОГОВЫЙ СКОР:
clip_score = (
  0.20 * norm(hook) +
  0.15 * norm(pace) +
  0.15 * norm(intensity) +
  0.15 * norm(clarity) +
  0.15 * norm(emotion) +
  0.10 * norm(boundary) +
  0.10 * norm(momentum)
)

ВСЕ ПОДСКОРЫ НОРМАЛИЗУЮТСЯ В [0, 1]

ДИВЕРСИФИКАЦИЯ:
1. Сортируем кандидаты по clip_score DESC
2. Жадный выбор top-N (например N=50):
   for each candidate:
     overlap_with_selected = max(overlap_ratios)
     if overlap_with_selected > 0.65:
       adjusted_score = clip_score * 0.3
     elif overlap_with_selected > 0.35:
       adjusted_score = clip_score * 0.7
     else:
       adjusted_score = clip_score
     
     if adjusted_score still in top-N:
       SELECT candidate
       
3. LOCAL RE-FIT (опционально):
   if overlap > 0.65 with selected:
     try shifts: [-8, -5, -3, -2, +2, +3, +5, +8] sec
     choose shift with max(adjusted_score)

Выход: список клипов с score breakdown + reasons (строки)
```

---

### Слой 6: Контекст и описание (Context Layer) — ОПЦИОНАЛЬНО

#### **6.1. TopicSegmentationHandler** (опционально)
```
Вход: transcriptSegments, timeline
Действие: выделение тематических блоков (смены темы в транскрипте)
Выход: список topic_segments {start, end, topic}
```

#### **6.2. BlockAnalysisHandler** (опционально, с LLM)
```
Вход: сегменты речи, сгруппированные по темам
Действие: для каждого блока LLM обобщает содержание
Выход: block descriptions
Важно: батчирование, request pooling
```

#### **6.3. LLMRefineCandidatesHandler** (опционально)
```
Вход: top-30 клипов из 5.2
Действие: для каждого клипа LLM проверяет:
  - "Понятен ли этот момент без контекста?"
  - "Есть ли в нём ясное начало и конец?"
  - Уточняет title/description
Выход: refined_clips с LLM-проверкой
Важно: параллелизм (батчи), кэширование, timeouts
```

---

### Слой 7: Финализация (Finalization Layer)

#### **7.1. ChapterBuilderHandler**
```
Вход: timeline, highlights, scenes, transcript
Действие: разбиение видео на главы по сценам + тематическим переходам
Выход: chapters {start, end, title, description}
```

#### **7.2. FinalizeAnalysisHandler**
```
Вход: всё состояние анализа
Действие:
  1. Сборка AnalysisResult
  2. Сериализация в JSON (с numpy encoding)
  3. Компрессия (опционально gzip)
  4. Сохранение в output_dir
Выход: путь к JSON-файлу
```

---

## Новая архитектура обработки состояния

### Вместо линейной цепи → DAG с параллелизмом

```
                    ┌─ MotionAnalysisHandler ─┐
                    │                         │
ReadFile ──┬─ VideoMeta ─┤─ AudioExtraction ─┤
           │            ├─ SceneDetection ──┤
           └─ FFmpeg────┤─ AudioFeatures ────┼─→ FusionTimeline
                        │ AudioEvents ───────┤
                        │ SpeechToText ──────┤
                        │ Sentiment ─────────┤
                        │ SpeechQuality ─────┤
                        └─ HumorDetection ───┘
                        
FusionTimeline → CandidateSelection → HighlightDetection → [LLMRefine?] → ChapterBuilder → Finalize
```

### State Management

```python
@dataclass
class AnalysisState:
    analysisId: str
    videoPath: str
    globalConfig: GlobalConfig
    
    # Completed stages (marked as completed after each handler)
    completed_stages: Set[str]  # {'ReadFile', 'VideoMeta', ...}
    
    # Results from each handler
    videoMetadata: Optional[VideoMetadata] = None
    audioPath: Optional[str] = None
    motionAnalysis: Optional[MotionAnalysis] = None
    audioFeatures: Optional[AudioFeatures] = None
    audioEvents: Optional[AudioEventsResult] = None
    sceneDetection: Optional[SceneDetectionResult] = None
    speechToText: Optional[SpeechToTextResult] = None
    sentiment: Optional[SentimentAnalysisResult] = None
    speechQuality: Optional[SpeechQualityResult] = None
    timeline: Optional[TimelineResult] = None
    candidates: Optional[List[Candidate]] = None
    highlights: Optional[HighlightDetectionResult] = None
    chapters: Optional[ChapterBuilderResult] = None
    
    # Status
    status: Literal["Pending", "InProgress", "Completed", "Failed"]
    currentStage: Optional[str] = None
    error: Optional[str] = None
    startTime: float = field(default_factory=time.time)
    endTime: Optional[float] = None
```

---

## Параллелизм и производительность

### Уровень 1: Параллельные обработчики (Слой 2)
```python
# Все эти работают одновременно на разных процессах/потоках
handlers = [
    MotionAnalysisHandler(),
    AudioExtractionHandler(),
    SceneDetectionHandler(),
    AudioFeaturesHandler(),
    AudioEventsHandler(),
]

# Параллельное выполнение
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(h.process, state) for h in handlers]
    results = [f.result() for f in futures]
```

### Уровень 2: Батчирование для ML моделей
```python
# STT, Sentiment, LLM — все могут работать с батчами
transcripts = split_to_batches(segments, batch_size=32)
for batch in transcripts:
    sentiment_scores = model.predict_batch(batch)
```

### Уровень 3: GPU использование
```python
# CUDA detection определяет:
if GLOBAL_CONFIG.device == "cuda":
    model = model.to("cuda")
    batch_size = 32  # больше батч для GPU
else:
    batch_size = 8   # меньше для CPU
```

---

## Переработанный пайплайн: Минimalный vs Полный

### Минimalный пайплайн (для тестирования, 5 мин на 60-мин видео):
```
ReadFile → VideoMeta → MotionAnalysis ┐
AudioExtraction → AudioFeatures       ├→ FusionTimeline → HighlightDetection → Finalize
SceneDetection ───────────────────────┘
```

### Полный пайплайн (with LLM refinement, 15 мин на 60-мин видео):
```
[Layer 1: Init] → [Layer 2: Fast (parallel)] → [Layer 3: Medium (parallel)] →
[Layer 4: Fusion] → [Layer 5: Highlights] → [Layer 6: Context (optional)] →
[Layer 7: Finalize]
```

---

## Сравнение с предыдущим подходом

| Аспект | Старый подход | Новый подход |
|--------|---------------|--------------|
| **Архитектура** | Линейная цепь | DAG с параллелизмом |
| **Скоринг** | Один `topinterest` | Многоуровневый (hook, pace, intensity, ...) |
| **LLM** | Для всех моментов | Только для top-N (опционально) |
| **Якоря** | Только interest пики | 6 типов якорей |
| **Диверсификация** | Автоматическое удаление overlap | Штраф за overlap + local re-fit |
| **Параллелизм** | Нет | Полный для слоя 2–3 |
| **Производительность** | 30 мин на 1 час видео | ~10–15 мин на 1 час видео |

---

## Ключевые изменения в коде

1. **DAG Executor** — новый модуль для управления параллелизмом (использовать `asyncio` или `ThreadPoolExecutor`)
2. **CandidateSelectionHandler** — полностью новый хэндлер с якорями
3. **HighlightDetectionHandler** — переписать с новым скорингом и диверсификацией
4. **State management** — добавить tracking `completed_stages`
5. **Конфиг** — добавить параметры параллелизма (num_workers, batch_sizes)

---

## Финальная рекомендация по архитектуре

✅ **Использовать DAG (Directed Acyclic Graph) для пайплайна**
- Позволяет параллелизм
- Легко добавлять новые узлы
- Удобно отлаживать (видно состояние каждого узла)

✅ **Многоуровневый скоринг (не один interest)**
- Универсален для любого контента
- Объяснимый (score breakdown)

✅ **LLM только для refinement, не для основного анализа**
- Быстро (только top-N)
- Экономит API calls

✅ **Параллелизм на уровне обработчиков**
- 3–5x ускорение
- Асинхронная обработка

---

Это переосмысленный пайплайн, который сочетает лучшие практики из VideoPipe (DAG), PreenCut (LLM refinement), и собственные улучшения (многоуровневый скоринг, диверсификация).
