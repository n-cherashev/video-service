# Video Processing Pipeline - Техническое задание

## 📚 Список задач

Проект разделен на отдельные задачи. Каждая задача описана в своем файле:

### Вводные задачи (обязательные для начала)

1. **TASK_0.md** — Подготовка проекта
   - Создание директорий и базовых файлов
   - Конфигурация проекта

2. **TASK_1.md** — Абстрактный базовый класс BaseHandler
   - Создание интерфейса для всех handlers
   - Использование ABC и abstractmethod

3. **TASK_2.md** — Первый handler: ReadFileHandler
   - Чтение и валидация видеофайла
   - Обработка ошибок

---

## 🎯 Общая архитектура

### Контекст (dict[str, Any])

Общий словарь, передаваемый между handlers:

```python
context = {
    "input_path": "videos/sample.mp4",      # Начальные данные
    "video_path": "/full/path/to/video.mp4",  # Добавлено ReadFileHandler
    "video_size_bytes": 12345,                  # Добавлено ReadFileHandler
    # ... результаты других handlers
}
```

### Handler (класс)

Каждый handler наследует `BaseHandler` и реализует метод `handle()`:

```python
class SomeHandler(BaseHandler):
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        # Получаем необходимые данные из контекста
        # Выполняем обработку
        # Добавляем результаты в контекст
        # Возвращаем контекст
        return context
```

### Пайплайн (main.py)

Главный скрипт запускает handlers по очереди:

```python
handlers = [
    ReadFileHandler(),
    # Здесь будут другие handlers
]

for handler in handlers:
    context = handler.handle(context)
```

---

## 📝 Требования к коду

### 1. Типизация (ОБЯЗАТЕЛЬНО!)

```python
# ✅ ПРАВИЛЬНО
from typing import Any

def handle(self, context: dict[str, Any]) -> dict[str, Any]:
    pass

def __init__(self, param: str = "default") -> None:
    self.param = param

# ❌ НЕПРАВИЛЬНО
def handle(self, context: dict):  # Нет типов значений!
    pass

def __init__(self, param: str = "default"):  # Нет -> None!
    pass
```

### 2. Наследование

```python
# ✅ ПРАВИЛЬНО
from handlers.base_handler import BaseHandler

class MyHandler(BaseHandler):
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        # реализация
        return context
```

### 3. Обработка ошибок

```python
# ✅ ПРАВИЛЬНО
if not input_path:
    raise ValueError("'input_path' not provided in context")
```

### 4. Логирование

```python
# ✅ ПРАВИЛЬНО
print(f"✓ Action completed: {result}")
```

---

## 📊 Статус проекта

| Задача | Файл | Описание | Статус |
|--------|------|---------|--------|
| 0 | TASK_0.md | Подготовка проекта | ⏭️ |
| 1 | TASK_1.md | BaseHandler | ⏭️ |
| 2 | TASK_2.md | ReadFileHandler | ⏭️ |

---

## ✅ Как начать

1. Откройте **TASK_0.md**
2. Выполните все шаги
3. Откройте **TASK_1.md**
4. Реализуйте `BaseHandler`
5. Откройте **TASK_2.md**
6. Реализуйте `ReadFileHandler`

После выполнения всех задач структура проекта должна быть готова!

Создать простой **консольный пайплайн для обработки видео** с архитектурой на основе обработчиков (handlers).

**Основная идея:**
- Видеофайл поступает на вход
- Проходит через серию обработчиков (handlers)
- Каждый обработчик отвечает за один этап (чтение файла, извлечение аудио, анализ и т.д.)
- Данные между обработчиками передаются через общий контекст (`dict`)
- На выходе получаем результаты обработки (аудио, анализ движения, субтитры и т.д.)

## 📋 Архитектура проекта

```
video-service/
├── handlers/                          # Пакет с обработчиками
│   ├── __init__.py                   # Инициализация пакета
│   ├── base_handler.py               # Абстрактный базовый класс
│   ├── read_file_handler.py          # Чтение и валидация файла
│   ├── ffmpeg_extract_handler.py     # Извлечение аудио
│   ├── motion_analysis_handler.py    # Анализ движения (заглушка)
│   ├── audio_analysis_handler.py     # Анализ аудио (заглушка)
│   ├── subtitles_handler.py          # Извлечение субтитров (заглушка)
│   └── fusion_handler.py             # Сборка результатов (заглушка)
├── main.py                            # Точка входа, запуск пайплайна
├── pyproject.toml                     # Конфигурация проекта
├── .gitignore                         # Игнорируемые файлы
├── README.md                          # Описание проекта
└── task.md                            # Это файл (ТЗ)
```

## ⚙️ Принцип работы пайплайна

1. **Инициализация контекста**: `context = {"input_path": "videos/sample.mp4"}`
2. **Создание списка handlers**: `[ReadFileHandler(), FFmpegExtractHandler(), ...]`
3. **Последовательный запуск**:
   ```python
   for handler in handlers:
       context = handler.handle(context)  # Каждый handler преобразует контекст
   ```
4. **Вывод результатов**: Финальный контекст содержит все результаты обработки

## 📝 Требования к коду

1. ✅ **ОБЯЗАТЕЛЬНАЯ ТИПИЗАЦИЯ** — все функции/методы должны иметь аннотации типов
2. ✅ **Наследование от BaseHandler** — все handlers должны быть потомками `BaseHandler`
3. ✅ **Обработка ошибок** — понятные исключения с описанием
4. ✅ **Логирование** — вывод информации о ходе выполнения через `print()`

## 📌 Требования к типизации

### Обязательные правила

```python
# ✅ ПРАВИЛЬНО
from typing import Any

def handle(self, context: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    return context

def __init__(self, param: str = "default") -> None:
    self.param = param
```

```python
# ❌ НЕПРАВИЛЬНО
def handle(self, context: dict):  # Без типов значений!
    context = {}                   # Без аннотации типа!
    return context

def __init__(self, param: str = "default"):  # Без -> None!
    self.param = param
```

---

## 📋 Задача 1. Подготовка каркаса проекта

**Статус**: ⏭️ Требует реализации

### 1.1 Создать директорию `handlers/`

```bash
mkdir -p handlers
```

### 1.2 Создать файл `handlers/__init__.py`

Это файл инициализации Python пакета. Здесь импортируются все handlers для удобного доступа.

**Файл**: `handlers/__init__.py`

```python
"""Video processing handlers package."""

from handlers.base_handler import BaseHandler
from handlers.read_file_handler import ReadFileHandler
from handlers.ffmpeg_extract_handler import FFmpegExtractHandler
from handlers.motion_analysis_handler import MotionAnalysisHandler
from handlers.audio_analysis_handler import AudioAnalysisHandler
from handlers.subtitles_handler import SubtitlesHandler
from handlers.fusion_handler import FusionHandler

__all__ = [
    "BaseHandler",
    "ReadFileHandler",
    "FFmpegExtractHandler",
    "MotionAnalysisHandler",
    "AudioAnalysisHandler",
    "SubtitlesHandler",
    "FusionHandler",
]
```

### 1.3 Создать файл `pyproject.toml`

**Файл**: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "video-service"
version = "0.1.0"
description = "Console pipeline for video processing with handler-based architecture"
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
dev = []
```

### 1.4 Создать файл `.gitignore`

**Файл**: `.gitignore`

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyCharm
.idea/
*.swp
*.swo

# VS Code
.vscode/

# Temporary files
temp/
*.tmp
*.log

# OS
.DS_Store
```

### 1.5 Создать файл `README.md`

**Файл**: `README.md`

```markdown
# Video Service

Консольный пайплайн для обработки видео с архитектурой на основе обработчиков (handlers).
Каждый этап обработки видеофайла реализован отдельным классом-обработчиком, которые
последовательно вызываются через общий контекст.

## Запуск

\`\`\`bash
python main.py
\`\`\`
```

---

## 📋 Задача 2. Создать абстрактный базовый класс BaseHandler

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/base_handler.py`

### Описание

Это абстрактный базовый класс, который определяет **интерфейс** для всех handlers.
Все обработчики должны от него наследовать и реализовать метод `handle()`.

### Полная реализация

```python
from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """
    Абстрактный базовый класс для всех handlers в пайплайне.

    Все обработчики должны наследовать этот класс и реализовать
    метод handle(), который преобразует контекст.
    """

    @abstractmethod
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Абстрактный метод для обработки контекста.

        Каждый handler должен переопределить этот метод и реализовать
        свою логику обработки.

        Args:
            context: Словарь с данными пайплайна, передаваемый между handlers

        Returns:
            Обновленный контекст с результатами обработки

        Raises:
            Различные исключения в зависимости от реализации handler-а
        """
        pass
```

### Как это работает

**Импорты**:
- `from abc import ABC, abstractmethod` — для создания абстрактного класса
- `from typing import Any` — для типизации

**Класс BaseHandler(ABC)**:
- `ABC` = Abstract Base Class (абстрактный базовый класс)
- Это значит, что класс нельзя инстанцировать напрямую: `BaseHandler()` вызовет ошибку

**Метод @abstractmethod**:
- Обозначает, что это абстрактный метод
- Все потомки **ОБЯЗАНЫ** переопределить этот метод
- Без переопределения создать экземпляр потомка не получится

### Пример использования

```python
# ❌ ОШИБКА: Нельзя создать экземпляр абстрактного класса
handler = BaseHandler()
# TypeError: Can't instantiate abstract class BaseHandler with abstract method handle

# ❌ ОШИБКА: Потомок не реализовал handle()
class IncompleteHandler(BaseHandler):
    pass

handler = IncompleteHandler()
# TypeError: Can't instantiate abstract class IncompleteHandler with abstract method handle

# ✅ ПРАВИЛЬНО: Полная реализация handle()
class CompleteHandler(BaseHandler):
    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

handler = CompleteHandler()  # Работает!

---

## 📋 Задача 3. Реализовать ReadFileHandler

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/read_file_handler.py`

### Описание

Первый обработчик в пайплайне. Его задача — прочитать путь к видеофайлу из контекста,
проверить, что файл существует, и добавить информацию о файле в контекст.

### Полная реализация

```python
import os
from pathlib import Path
from typing import Any

from handlers.base_handler import BaseHandler


class ReadFileHandler(BaseHandler):
    """Handler for reading and validating video file."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Read and validate video file from context.

        Args:
            context: Dictionary with 'input_path' key containing path to video file.

        Returns:
            Updated context with 'video_path' and 'video_size_bytes'.

        Raises:
            ValueError: If 'input_path' not provided in context.
            FileNotFoundError: If file doesn't exist.
            IsADirectoryError: If path is a directory, not a file.
        """
        # Получаем путь из контекста
        input_path = context.get("input_path")
        if not input_path:
            raise ValueError("'input_path' not provided in context")

        # Нормализуем путь (преобразуем в абсолютный)
        video_path = str(Path(input_path).resolve())

        # Проверяем, что файл существует
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Проверяем, что это файл, а не директория
        if not os.path.isfile(video_path):
            raise IsADirectoryError(f"Path is not a file: {video_path}")

        # Получаем размер файла в байтах
        video_size_bytes = os.path.getsize(video_path)

        # Добавляем информацию в контекст
        context["video_path"] = video_path
        context["video_size_bytes"] = video_size_bytes

        # Выводим информацию о выполнении
        print(f"✓ File read: {video_path} ({video_size_bytes} bytes)")

        return context
```

### Как это работает

1. **Получить путь**: `input_path = context.get("input_path")`
2. **Валидация**: Проверяем, что путь существует и это файл
3. **Получить размер**: `os.path.getsize(video_path)`
4. **Добавить в контекст**: Два новых ключа: `video_path` и `video_size_bytes`
5. **Вернуть контекст**: `return context`

### Тестирование

```python
# Создаем контекст с путем к видеофайлу
context = {"input_path": "videos/sample.mp4"}

# Создаем handler и запускаем его
handler = ReadFileHandler()
result = handler.handle(context)

# Проверяем результат
assert "video_path" in result
assert "video_size_bytes" in result
print(result)
# {'input_path': 'videos/sample.mp4', 'video_path': '/full/path/to/videos/sample.mp4', 'video_size_bytes': 12345}
```

---

## 📋 Задача 4. Реализовать FFmpegExtractHandler

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/ffmpeg_extract_handler.py`

### Описание

Второй обработчик в пайплайне. Его задача — извлечь аудиодорожку из видео
с помощью утилиты FFmpeg и сохранить её во временную папку.

### Полная реализация

```python
import os
import subprocess
from pathlib import Path
from typing import Any

from handlers.base_handler import BaseHandler


class FFmpegExtractHandler(BaseHandler):
    """Handler for extracting audio from video using FFmpeg."""

    def __init__(self, temp_dir: str = "temp") -> None:
        """
        Initialize FFmpeg handler.

        Args:
            temp_dir: Directory to store temporary audio files.
        """
        self.temp_dir = temp_dir

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract audio from video file using FFmpeg.

        Args:
            context: Dictionary with 'video_path' key.

        Returns:
            Updated context with 'audio_path'.

        Raises:
            ValueError: If 'video_path' not provided in context.
            RuntimeError: If FFmpeg extraction fails.
        """
        # Получаем путь к видео из контекста
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        # Создаем временную директорию, если её нет
        Path(self.temp_dir).mkdir(exist_ok=True)

        # Генерируем путь для выходного аудиофайла
        video_name = Path(video_path).stem  # Имя файла без расширения
        audio_path = os.path.join(self.temp_dir, f"{video_name}.wav")

        # Подготавливаем команду FFmpeg
        cmd = [
            "ffmpeg",
            "-i", video_path,           # Входной файл
            "-q:a", "9",                # Качество аудио
            "-n",                       # Не перезаписывать существующие файлы
            audio_path                  # Выходной файл
        ]

        # Запускаем FFmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install ffmpeg: brew install ffmpeg"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr.decode()}")

        # Добавляем путь к аудио в контекст
        context["audio_path"] = audio_path

        # Выводим информацию о выполнении
        print(f"✓ Audio extracted: {audio_path}")

        return context
```

### Как это работает

1. **Конструктор**: Сохраняет директорию для временных файлов
2. **Получить видео**: `video_path = context.get("video_path")`
3. **Создать папку**: `Path(self.temp_dir).mkdir(exist_ok=True)`
4. **Сгенерировать имя**: Берем имя видео и добавляем расширение `.wav`
5. **Запустить FFmpeg**: Через `subprocess.run()` с параметрами
6. **Добавить в контекст**: `context["audio_path"] = audio_path`
7. **Вернуть контекст**: `return context`

### Требования

**Установка FFmpeg на macOS**:
```bash
brew install ffmpeg
```

**Параметры FFmpeg**:
- `-i` — входной файл
- `-q:a 9` — качество аудио (9 = лучшее качество)
- `-n` — не перезаписывать файлы

### Тестирование

```python
context = {
    "video_path": "/full/path/to/videos/sample.mp4"
}

handler = FFmpegExtractHandler()
result = handler.handle(context)

assert "audio_path" in result
print(result["audio_path"])
# temp/sample.wav
```

---

## 📋 Задача 5. Реализовать MotionAnalysisHandler (заглушка)

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/motion_analysis_handler.py`

### Описание

Заглушка для анализа движения в видео. На данный момент просто добавляет
фиктивное значение в контекст без реального анализа (это можно реализовать позже с OpenCV).

### Полная реализация

```python
from typing import Any

from handlers.base_handler import BaseHandler


class MotionAnalysisHandler(BaseHandler):
    """Handler for analyzing motion in video (stub)."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze motion in video (stub implementation).

        Args:
            context: Dictionary with 'video_path' key.

        Returns:
            Updated context with 'motion_score' set to 0.0.

        Raises:
            ValueError: If 'video_path' not provided in context.
        """
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("'video_path' not provided in context")

        context["motion_score"] = 0.0

        print("✓ Motion analysis done (stub)")

        return context
```

---

## 📋 Задача 6. Реализовать AudioAnalysisHandler (заглушка)

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/audio_analysis_handler.py`

### Описание

Заглушка для анализа аудио. На данный момент просто добавляет фиктивное значение
в контекст без реального анализа.

### Полная реализация

```python
from typing import Any

from handlers.base_handler import BaseHandler


class AudioAnalysisHandler(BaseHandler):
    """Handler for analyzing audio (stub)."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze audio (stub implementation).

        Args:
            context: Dictionary with 'audio_path' key.

        Returns:
            Updated context with 'audio_energy' set to 0.0.

        Raises:
            ValueError: If 'audio_path' not provided in context.
        """
        audio_path = context.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' not provided in context")

        context["audio_energy"] = 0.0

        print("✓ Audio analysis done (stub)")

        return context
```

---

## 📋 Задача 7. Реализовать SubtitlesHandler (заглушка)

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/subtitles_handler.py`

### Описание

Заглушка для извлечения субтитров. На данный момент просто добавляет пустой список
в контекст без реального анализа речи (это можно реализовать позже с помощью STT).

### Полная реализация

```python
from typing import Any

from handlers.base_handler import BaseHandler


class SubtitlesHandler(BaseHandler):
    """Handler for extracting subtitles (stub)."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract subtitles from audio (stub implementation).

        Args:
            context: Dictionary with 'audio_path' key.

        Returns:
            Updated context with 'subtitles' set to empty list.

        Raises:
            ValueError: If 'audio_path' not provided in context.
        """
        audio_path = context.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' not provided in context")

        context["subtitles"] = []

        print("✓ Subtitles extracted (stub)")

        return context
```

---

## 📋 Задача 8. Реализовать FusionHandler (заглушка)

**Статус**: ⏭️ Требует реализации

**Файл**: `handlers/fusion_handler.py`

### Описание

Заглушка для сборки финального таймлайна и тепловой карты. На данный момент
просто добавляет пустые списки в контекст.

### Полная реализация

```python
from typing import Any

from handlers.base_handler import BaseHandler


class FusionHandler(BaseHandler):
    """Handler for building timeline and heatmap (stub)."""

    def handle(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Build timeline and heatmap from analysis results (stub).

        Args:
            context: Dictionary with 'motion_score', 'audio_energy', 'subtitles'.

        Returns:
            Updated context with 'timeline' and 'heatmap' added.

        Raises:
            ValueError: If required keys not provided in context.
        """
        if "motion_score" not in context:
            raise ValueError("'motion_score' not provided in context")
        if "audio_energy" not in context:
            raise ValueError("'audio_energy' not provided in context")
        if "subtitles" not in context:
            raise ValueError("'subtitles' not provided in context")

        context["timeline"] = []
        context["heatmap"] = []

        print("✓ Timeline and heatmap built (stub)")

        return context
```

---

## 📋 Задача 9. Реализовать main.py

**Статус**: ⏭️ Требует реализации

**Файл**: `main.py`

### Описание

Главный файл проекта, который:
1. Инициализирует контекст с путем к видеофайлу
2. Создает список всех handlers в нужном порядке
3. Запускает их по очереди
4. Выводит финальный результат

### Полная реализация

```python
#!/usr/bin/env python3
"""
Main entry point for video processing pipeline.

Usage:
    python main.py
"""

from typing import Any

from handlers.base_handler import BaseHandler
from handlers.ffmpeg_extract_handler import FFmpegExtractHandler
from handlers.motion_analysis_handler import MotionAnalysisHandler
from handlers.audio_analysis_handler import AudioAnalysisHandler
from handlers.read_file_handler import ReadFileHandler
from handlers.subtitles_handler import SubtitlesHandler
from handlers.fusion_handler import FusionHandler


def main() -> None:
    """
    Run the video processing pipeline.

    Порядок выполнения handlers:
    1. ReadFileHandler — чтение и валидация файла
    2. FFmpegExtractHandler — извлечение аудио
    3. MotionAnalysisHandler — анализ движения
    4. AudioAnalysisHandler — анализ аудио
    5. SubtitlesHandler — извлечение субтитров
    6. FusionHandler — сборка результатов
    """
    video_path = "videos/sample.mp4"
    context: dict[str, Any] = {"input_path": video_path}

    handlers: list[BaseHandler] = [
        ReadFileHandler(),
        FFmpegExtractHandler(),
        MotionAnalysisHandler(),
        AudioAnalysisHandler(),
        SubtitlesHandler(),
        FusionHandler(),
    ]

    print("Starting video processing pipeline...\n")
    try:
        for handler in handlers:
            context = handler.handle(context)
        print("\n✓ Pipeline completed successfully")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        return

    print("\nFinal context:")
    for key, value in context.items():
        if key != "input_path":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

---

## 🎯 Резюме

| Задача | Файл | Статус |
|--------|------|--------|
| 1 | `handlers/__init__.py`, `pyproject.toml`, `.gitignore`, `README.md` | ⏭️ |
| 2 | `handlers/base_handler.py` | ⏭️ |
| 3 | `handlers/read_file_handler.py` | ⏭️ |
| 4 | `handlers/ffmpeg_extract_handler.py` | ⏭️ |
| 5 | `handlers/motion_analysis_handler.py` | ⏭️ |
| 6 | `handlers/audio_analysis_handler.py` | ⏭️ |
| 7 | `handlers/subtitles_handler.py` | ⏭️ |
| 8 | `handlers/fusion_handler.py` | ⏭️ |
| 9 | `main.py` | ⏭️ |

---

## ✅ Контрольный список

При выполнении каждой задачи проверьте:

- [ ] Все файлы созданы
- [ ] Все функции и методы имеют аннотации типов
- [ ] Используется `dict[str, Any]` вместо просто `dict`
- [ ] Все конструкторы имеют `-> None`
- [ ] Все handlers наследуют `BaseHandler`
- [ ] Обработаны ошибки с понятными исключениями
- [ ] Есть информационные выводы `print()`
- [ ] `python main.py` запускается без ошибок
- [ ] Вывод соответствует ожидаемому
