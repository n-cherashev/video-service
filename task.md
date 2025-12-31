Сейчас проект уже структурирован вокруг пайплайна хэндлеров и общего dict‑контекста, и из контекста видно, какие сущности и фичи реально живут в системе: motion, audio, speech_to_text, sentiment, humor, topics, timeline, highlights, chapters, analysis_result.[1]

Ниже — предложение по новой структуре и план миграции.

## Цели структуры

- **Фичи сгруппированы** по смыслу (аудио, видео, NLP, интеграция сигналов).
- **Модели отделены** от хэндлеров и I/O.
- **Пайплайн плоский** и не знает деталей фич, только обрабатывает `PipelineState`.
- **Логи и консольный вывод** инкапсулированы в хэндлерах, а не в `main.py`.[1]

***

## Предлагаемое дерево папок

Базовый уровень:

- `config/`
- `core/` — базовые абстракции и пайплайн.
- `features/` — доменные фичи (motion, audio, nlp, fusion, highlights, chapters).
- `models/` — общие доменные модели (VideoInfo, TimelinePoint, Highlight, Chapter, AnalysisResult, и т.п.).
- `io/` — чистый I/O (ffmpeg, файловая система, cli).
- `utils/` — мелкие утилиты (timeseries, truncation, serde и т.д.).[1]

Пример дерева (упрощённо):

```text
video-service/
  main.py
  config/
    __init__.py
    settings.py        # VideoServiceSettings, build_initial_context -> build_initial_state
  core/
    __init__.py
    base_handler.py    # BaseHandler
    pipeline.py        # run_pipeline(state, handlers)
    logging.py         # типы логов/форматтер (если понадобится)
  models/
    __init__.py
    common.py          # VideoInfo, Scene, TopicSegment, AudioEvent, etc.
    timeline.py        # TimelinePoint
    highlights.py      # Highlight, HighlightType
    chapters.py        # Chapter
    analysis.py        # AnalysisResult (агрегирует всё для REST/CLI)
    state.py           # PipelineState (центральная модель состояния)
    serde.py           # to_jsonable() и, возможно, from_json()
  features/
    file_io/
      read_file_handler.py
      video_meta_handler.py
      ffmpeg_extract_handler.py
    motion/
      motion_analysis_frame_diff_handler.py
      # позже: другие варианты детекции движения
    audio/
      audio_features_handler.py
      audio_events_handler.py
    scenes/
      scene_detection_handler.py
    nlp/
      speech_to_text_handler.py
      sentiment_analysis_handler.py
      humor_detection_handler.py
      topic_segmentation_handler.py
    fusion/
      fusion_timeline_handler.py
    highlights/
      highlight_detection_handler.py
    chapters/
      chapter_builder_handler.py
    finalize/
      finalize_analysis_handler.py
  io/
    cli.py             # парсинг аргументов, запуск main
    ffmpeg.py          # обёртки над ffmpeg, если захочешь вынести
  utils/
    __init__.py
    timeseries.py      # Series, extract_series, interp_to_grid, normalize_01
    truncation.py      # truncate_large_lists
```

Плюс файл `pyproject.toml` / `setup.cfg` по желанию — чтобы импортировать как пакет, но это опционально.

***

## Модели: центральный PipelineState

Сейчас `Context` — это голый dict, который разрастается; при этом лог уже показывает логически связанные кластеры (`analysis_result` с вложенными разделами).  Это хороший черновик для будущих моделей.[1]

### 1. Общие сущности

`models/common.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    size_bytes: int
    fps: float
    duration_seconds: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class Scene:
    index: int
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TopicSegment:
    start: float
    end: float
    topic: str


@dataclass(frozen=True, slots=True)
class AudioEvent:
    start: float
    end: float
    time: float
    type: str
    confidence: float
    peak_value: float
```

### 2. Уже начатые модели

`models/timeline.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    time: float
    interest: float
    motion: float
    audio_loudness: float
    sentiment: float
    humor: float
    has_laughter: bool
    has_loud_sound: bool
    is_scene_boundary: bool
    is_dialogue: bool
```

`models/highlights.py`:

```python
from dataclasses import dataclass
from typing import Literal


HighlightType = Literal["top_interest", "rules"]


@dataclass(frozen=True, slots=True)
class Highlight:
    start: float
    end: float
    type: HighlightType
    score: float
```

`models/chapters.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chapter:
    start: float
    end: float
    title: str
    description: str
```

### 3. PipelineState — “истина” для пайплайна

`models/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .common import VideoInfo, Scene, TopicSegment, AudioEvent
from .timeline import TimelinePoint
from .highlights import Highlight
from .chapters import Chapter


@dataclass
class PipelineState:
    settings: Any                      # VideoServiceSettings или более строгий тип
    video_info: Optional[VideoInfo] = None

    # Сырые данные
    input_path: Optional[str] = None
    audio_path: Optional[str] = None

    motion_heatmap: list[dict] = field(default_factory=list)
    audio_features: dict[str, list[dict]] = field(default_factory=dict)
    audio_events: list[AudioEvent] = field(default_factory=list)

    scenes: list[Scene] = field(default_factory=list)
    scene_boundaries: list[float] = field(default_factory=list)

    transcript_segments: list[dict] = field(default_factory=list)
    full_transcript: Optional[str] = None

    sentiment_timeline: list[dict] = field(default_factory=list)
    humor_scores: list[dict] = field(default_factory=list)
    topic_segments: list[TopicSegment] = field(default_factory=list)

    timeline_points: list[TimelinePoint] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)

    # summary/diagnostics, если нужно
    summaries: dict[str, Any] = field(default_factory=dict)
```

Параллельно можно сохранить “экспортный” `analysis_result`, но он скорее нужен как DTO наружу.

***

## Core: BaseHandler и PipelineRunner

`core/base_handler.py` — ты уже сделал почти идеальный базовый интерфейс:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """Base handler interface."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """Process context and return updated context."""
        raise NotImplementedError
```

План эволюции:

- Этап 1: `context: dict[str, Any]` (как сейчас) + параллельно `PipelineState` внутри.
- Этап 2: изменить сигнатуру `handle(self, state: PipelineState) -> PipelineState` и адаптировать `main.py`.  

Сначала можно сделать адаптер:

```python
# core/pipeline.py
from typing import Any, Iterable

from core.base_handler import BaseHandler


def run_pipeline(context: dict[str, Any], handlers: Iterable[BaseHandler]) -> dict[str, Any]:
    for i, h in enumerate(handlers, 1):
        print(f"[{i:2d}] {h.name}")
        context = h.handle(context)
    return context
```

Чуть позже: `run_pipeline(state: PipelineState, handlers: Iterable[PipelineHandler])`.

***

## features: группировка хэндлеров

Твой текущий список хэндлеров (по логам) уже соответствует фичам:[1]

1. ReadFileHandler  
2. VideoMetaHandler  
3. FFmpegExtractHandler  
4. MotionAnalysisFrameDiffHandler  
5. AudioFeaturesHandler  
6. AudioEventsHandler  
7. SceneDetectionHandler  
8. SpeechToTextHandler  
9. SentimentAnalysisHandler  
10. HumorDetectionHandler  
11. TopicSegmentationHandler  
12. FusionTimelineHandler  
13. HighlightDetectionHandler  
14. ChapterBuilderHandler  
15. FinalizeAnalysisHandler  

Предлагаемое размещение:

- `features/file_io/`
  - `read_file_handler.py`
  - `video_meta_handler.py`
  - `ffmpeg_extract_handler.py`
- `features/motion/`
  - `motion_analysis_frame_diff_handler.py`
- `features/audio/`
  - `audio_features_handler.py`
  - `audio_events_handler.py`
- `features/scenes/`
  - `scene_detection_handler.py`
- `features/nlp/`
  - `speech_to_text_handler.py`
  - `sentiment_analysis_handler.py`
  - `humor_detection_handler.py`
  - `topic_segmentation_handler.py`
- `features/fusion/`
  - `fusion_timeline_handler.py`
- `features/highlights/`
  - `highlight_detection_handler.py`
- `features/chapters/`
  - `chapter_builder_handler.py`
- `features/finalize/`
  - `finalize_analysis_handler.py`

Импорты везде обновятся с `from handlers.xxx` на `from features.xxx.yyy_handler import ...`.

***

## Логи: перенос из main в хэндлеры

Сейчас `main.py` печатает шаги пайплайна, а многие хэндлеры внутри тоже печатают свои summary (например, `✓ Audio events: 33 detected`, `✓ Timeline: 230 points, step=1.0s`).[1]

Идея:

- `core.pipeline.run_pipeline` отвечает только за общие “шаги” `[n/N] HandlerName`.
- Каждый хэндлер **сам** печатает свою короткую строку результата (`✓ Sentiment: 66 points` и т.п.).
- `main.py` только запускает пайплайн и печатает сводный SUMMARY + JSON.[1]

Пример внутри конкретного хэндлера (уже есть, просто зафиксировать стиль):

```python
class AudioEventsHandler(BaseHandler):
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        ...
        context["audio_events"] = events
        print(f"✓ Audio events: {len(events)} detected")
        return context
```

То же самое делаем для всех хэндлеров, где сейчас лог “зашит” в main.

---

## JSON-вывод: единый to_jsonable + truncate

У тебя уже есть огромный JSON в конце, плюс проблема с сериализацией `TimelinePoint`.[1]
Лучший вариант:

- `models/serde.py` с `to_jsonable` (dataclass -> dict/list/примитивы).
- В `main.py`: `jsonable_context = to_jsonable(context)` → `truncate_large_lists(jsonable_context)` → `json.dumps(...)`.[1]

Функция (мы уже набросали, повторю целиком):

```python
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    return obj
```

`main.py` (нижняя часть):

```python
from models.serde import to_jsonable
from utils.truncation import truncate_large_lists

...

print("\n" + "=" * 80)
print("FULL RESULTS (JSON)")
print("=" * 80)

jsonable_context = to_jsonable(context)
truncated_context = truncate_large_lists(jsonable_context)
print(json.dumps(truncated_context, indent=2, ensure_ascii=False))
```

Это зафиксирует контракт: внутри хэндлеров можно хранить **модели**, а на границе всё превращается в JSON-дружественные структуры.[1]

***

## План миграции по шагам

1) **Разнести файлы по папкам `core/`, `features/`, `models/`, `utils/`** без изменения логики.  
   - Обновить импорты в `main.py` и между модулями.[1]

2) **Ввести `models/timeline.py`, `models/highlights.py`, `models/chapters.py`, `models/common.py`, `models/serde.py`.**  
   - В `FusionTimelineHandler` добавить `timeline_points: list[TimelinePoint]` **параллельно** старому `timeline: list[dict]`.[1]
   - В `HighlightDetectionHandler` аналогично: `highlight_items: list[Highlight]` + `highlights` как dict для совместимости.

3) **Ввести `PipelineState` и адаптеры context ↔ state.**  
   - Пока `BaseHandler.handle` оставляем с `dict[str, Any]`, но `build_initial_context` может возвращать и dict, и `PipelineState`, а адаптер в `main.py` будет конвертировать.[1]

4) **Перенести все “✓ XYZ” принты в сами хэндлеры**, а `main.py` оставить только с `[n/N] HandlerName`, SUMMARY и JSON.

5) Когда большая часть фич перекочует на модели — поменять `BaseHandler` на:

```python
from models.state import PipelineState

class BaseHandler(ABC):
    @abstractmethod
    def handle(self, state: PipelineState) -> PipelineState:
        ...
```

и убрать dict‑контекст совсем.

---

Если хочешь, следующим шагом можно детально расписать конкретный рефакторинг для одной фичи (например, `fusion + highlights`): какие поля в `PipelineState`, какие модели, как именно переписать сигнатуры хэндлеров и логи.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/43060106/d344135f-3b54-4ab1-b31a-fe4a495bf6a6/paste.txt)