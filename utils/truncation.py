from typing import Any


def truncate_large_lists(obj: Any, max_items: int = 10) -> Any:
    if isinstance(obj, list):
        if len(obj) > max_items:
            first = obj[: max_items // 2]
            last = obj[-max_items // 2 :]
            return {
                "truncated": True,
                "total_length": len(obj),
                "first_items": first,
                "last_items": last,
            }
        return [truncate_large_lists(item, max_items) for item in obj]
    if isinstance(obj, dict):
        return {k: truncate_large_lists(v, max_items) for k, v in obj.items()}
    return obj