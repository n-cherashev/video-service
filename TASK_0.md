# Задача 1. Настройка проекта и первые handlers

**Статус**: ⏭️ Требует реализации

Это комплексная задача, которая включает:
1. Подготовку структуры проекта
2. Создание абстрактного базового класса `BaseHandler`
3. Реализацию первого handler-а `ReadFileHandler`

---

## Часть 1. Подготовка проекта

### 1.1 Создать директорию `handlers/`

```bash
mkdir -p handlers
```

### 1.2 Создать файл `handlers/__init__.py`

**Файл**: `handlers/__init__.py`

```python
"""Video processing handlers package."""

from handlers.base_handler import BaseHandler
from handlers.read_file_handler import ReadFileHandler

__all__ = [
    "BaseHandler",
    "ReadFileHandler",
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

## Часть 2. Абстрактный базовый класс BaseHandler

**Файл**: `handlers/base_handler.py`

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

### Объяснение

- **`ABC`** — Abstract Base Class делает класс абстрактным
- **`@abstractmethod`** — обозначает, что потомки обязаны реализовать этот метод
- **`dict[str, Any]`** — контекст: ключи всегда строки, значения любого типа

---

## Часть 3. Реализовать ReadFileHandler

**Файл**: `handlers/read_file_handler.py`

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
            Updated context with:
            - 'video_path': str — normalized path to the file
            - 'video_size_bytes': int — file size in bytes

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

1. **Получить путь** из контекста: `context.get("input_path")`
2. **Проверить валидность**: файл существует, это именно файл (не директория)
3. **Получить размер**: `os.path.getsize(video_path)`
4. **Добавить в контекст**: два новых ключа `video_path` и `video_size_bytes`
5. **Вернуть контекст**: `return context`

### Обработка ошибок

| Исключение | Когда | Пример |
|-----------|--------|---------|
| `ValueError` | `input_path` отсутствует | контекст `{}` |
| `FileNotFoundError` | файл не существует | путь `/nonexistent/file.mp4` |
| `IsADirectoryError` | путь — это директория | путь `/folder` |

---

## ✅ Контрольный список

При выполнении проверьте:

- [ ] Директория `handlers/` создана
- [ ] Файлы `__init__.py`, `base_handler.py`, `read_file_handler.py` созданы
- [ ] Файлы `pyproject.toml`, `.gitignore`, `README.md` созданы
- [ ] Все файлы имеют корректную типизацию (`dict[str, Any]`)
- [ ] `ReadFileHandler` наследует `BaseHandler`
- [ ] Все исключения имеют понятное описание
- [ ] Есть вывод через `print()`
- [ ] Импорты работают корректно
