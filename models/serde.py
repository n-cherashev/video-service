"""
Serialization utilities for models.

to_jsonable - конвертирует любые объекты в JSON-serializable структуры.
Поддерживает:
- dataclasses (с методом to_dict или через asdict)
- Enum (конвертирует в value)
- numpy types
- datetime
- set/frozenset
- pydantic BaseModel
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from enum import Enum
from typing import Any
import json


def to_jsonable(obj: Any) -> Any:
    """Convert any object to JSON-serializable types.

    Порядок проверок:
    1. None, bool, int, float, str - as is
    2. dict - recursive
    3. list/tuple - recursive
    4. set/frozenset - to sorted list
    5. dataclass with to_dict() - use it
    6. dataclass without to_dict() - use asdict
    7. Enum - use value
    8. datetime/date - to isoformat
    9. numpy types - convert to python
    10. pydantic BaseModel - model_dump
    11. fallback - str()
    """
    # Primitive types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # List/tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # Set/frozenset
    if isinstance(obj, (set, frozenset)):
        # Сортируем для детерминированного вывода
        try:
            return [to_jsonable(v) for v in sorted(obj)]
        except TypeError:
            # Если элементы несортируемые
            return [to_jsonable(v) for v in sorted(obj, key=str)]

    # Dataclass with to_dict method
    if is_dataclass(obj) and hasattr(obj, 'to_dict'):
        return to_jsonable(obj.to_dict())

    # Dataclass without to_dict
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    # Enum
    if isinstance(obj, Enum):
        return obj.value

    # Datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, date):
        return obj.isoformat()

    # Numpy types
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            return to_jsonable(obj.model_dump())
    except ImportError:
        pass

    # Object with to_dict method (generic)
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())

    # Object with __dict__
    if hasattr(obj, '__dict__'):
        return to_jsonable(vars(obj))

    # Fallback
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def from_json_file(path: str) -> Any:
    """Загружает JSON из файла."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_json_file(obj: Any, path: str, indent: int = 2) -> None:
    """Сохраняет объект в JSON файл."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(obj), f, indent=indent, ensure_ascii=False)


def to_json_str(obj: Any, indent: int | None = None) -> str:
    """Конвертирует объект в JSON строку."""
    return json.dumps(to_jsonable(obj), indent=indent, ensure_ascii=False)
