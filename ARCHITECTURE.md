# Новая архитектура Video Service

## Структура проекта

```
video-service/
├── main.py                    # Точка входа (обновлена)
├── config/                    # Конфигурация
│   ├── __init__.py
│   └── settings.py           # VideoServiceSettings, Context, build_initial_*
├── core/                     # Базовые абстракции
│   ├── __init__.py
│   ├── base_handler.py       # BaseHandler (перемещен из handlers/)
│   └── pipeline.py           # run_pipeline()
├── models/                   # Доменные модели
│   ├── __init__.py
│   ├── common.py            # VideoInfo, Scene, TopicSegment, AudioEvent
│   ├── timeline.py          # TimelinePoint
│   ├── highlights.py        # Highlight, HighlightType
│   ├── chapters.py          # Chapter
│   ├── state.py             # PipelineState (центральная модель)
│   └── serde.py             # to_jsonable() для JSON сериализации
├── features/                # Доменные фичи
│   ├── file_io/            # Работа с файлами
│   │   ├── read_file_handler.py
│   │   ├── video_meta_handler.py
│   │   └── ffmpeg_extract_handler.py
│   ├── motion/             # Анализ движения
│   │   └── motion_analysis_frame_diff_handler.py
│   ├── audio/              # Аудио анализ
│   │   ├── audio_features_handler.py
│   │   └── audio_events_handler.py
│   ├── scenes/             # Детекция сцен
│   │   └── scene_detection_handler.py
│   ├── nlp/                # NLP обработка
│   │   ├── speech_to_text_handler.py
│   │   ├── sentiment_analysis_handler.py
│   │   ├── humor_detection_handler.py
│   │   └── topic_segmentation_handler.py
│   ├── fusion/             # Интеграция сигналов
│   │   └── fusion_timeline_handler.py
│   ├── highlights/         # Выделение хайлайтов
│   │   └── highlight_detection_handler.py
│   ├── chapters/           # Формирование глав
│   │   └── chapter_builder_handler.py
│   └── finalize/           # Финализация результата
│       └── finalize_analysis_handler.py
├── utils/                  # Утилиты
│   ├── __init__.py
│   ├── timeseries.py       # Series, extract_series, interp_to_grid, normalize_01
│   └── truncation.py       # truncate_large_lists()
└── io/                     # I/O операции
    ├── __init__.py
    └── cli.py              # CLI парсинг
```

## Ключевые изменения

### 1. Группировка по доменам
- **features/file_io/**: Работа с файлами и метаданными
- **features/motion/**: Анализ движения в видео
- **features/audio/**: Аудио обработка и события
- **features/scenes/**: Детекция границ сцен
- **features/nlp/**: NLP задачи (STT, sentiment, humor, topics)
- **features/fusion/**: Интеграция всех сигналов в timeline
- **features/highlights/**: Выделение ключевых моментов
- **features/chapters/**: Создание глав для навигации

### 2. Центральные модели
- **PipelineState**: Основное состояние пайплайна (будущая замена dict-контекста)
- **TimelinePoint**: Типизированная точка временной шкалы
- **Highlight**: Структурированный хайлайт
- **Chapter**: Глава с заголовком и описанием

### 3. Улучшенная архитектура
- **core/pipeline.py**: Централизованный запуск пайплайна
- **models/serde.py**: Единая сериализация в JSON
- **utils/**: Переиспользуемые утилиты
- **config/**: Отдельная конфигурация

### 4. Логирование в хэндлерах
Каждый хэндлер теперь сам печатает свой результат:
```python
print(f"✓ Audio features: {len(times)} frames, sr={sr}Hz")
print(f"✓ Scene detection: {summary['count']} scenes")
```

## Миграция

### Этап 1: ✅ Структура папок и базовые модели
- Созданы все папки features/, core/, models/, utils/, io/
- Перемещены хэндлеры по доменным папкам
- Обновлены импорты

### Этап 2: Постепенное внедрение PipelineState
- Пока используется старый dict-контекст для совместимости
- PipelineState готов для будущего использования
- Адаптеры context ↔ state будут добавлены позже

### Этап 3: Типизированные модели
- TimelinePoint, Highlight, Chapter готовы к использованию
- FusionTimelineHandler может создавать timeline_points: list[TimelinePoint]
- Параллельно со старым timeline: list[dict] для совместимости

## Преимущества новой архитектуры

1. **Модульность**: Каждая фича изолирована в своей папке
2. **Типизация**: Строгие модели вместо dict
3. **Переиспользование**: Общие утилиты в utils/
4. **Расширяемость**: Легко добавлять новые фичи
5. **Тестируемость**: Каждый хэндлер можно тестировать отдельно
6. **Читаемость**: Четкое разделение ответственности

## Следующие шаги

1. Постепенная миграция на PipelineState
2. Внедрение типизированных моделей в хэндлерах
3. Добавление unit-тестов для каждой фичи
4. Создание REST API на основе новой архитектуры