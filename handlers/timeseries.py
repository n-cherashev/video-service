from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Series:
    """Time series: points with monotonic (non-decreasing) times."""
    times: np.ndarray  # shape (n,)
    values: np.ndarray  # shape (n,)

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.values.ndim != 1:
            raise ValueError("times/values must be 1D arrays")
        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError("times and values must have equal length")


def extract_series(
    points: Sequence[Mapping[str, object]],
    *,
    time_key: str = "time",
    value_key: str = "value",
    sort: bool = True,
) -> Series:
    if not points:
        return Series(times=np.array([], dtype=float), values=np.array([], dtype=float))

    times = []
    values = []
    for p in points:
        if time_key not in p or value_key not in p:
            continue
        try:
            t = float(p[time_key])  # type: ignore[arg-type]
            v = float(p[value_key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        times.append(t)
        values.append(v)

    if not times:
        return Series(times=np.array([], dtype=float), values=np.array([], dtype=float))

    t_arr = np.array(times, dtype=float)
    v_arr = np.array(values, dtype=float)

    if sort:
        idx = np.argsort(t_arr)
        t_arr = t_arr[idx]
        v_arr = v_arr[idx]

    return Series(times=t_arr, values=v_arr)


def interp_to_grid(series: Series, grid_times: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    if series.times.size == 0:
        return np.full_like(grid_times, fill_value, dtype=float)
    if series.times.size == 1:
        return np.full_like(grid_times, float(series.values[0]), dtype=float)

    # np.interp uses edge values outside by default; for analysis it's safer to clamp to fill.
    left = fill_value
    right = fill_value
    return np.interp(grid_times, series.times, series.values, left=left, right=right).astype(float)


def safe_mean_points(
    points: Sequence[Mapping[str, object]],
    *,
    value_key: str = "value",
) -> float:
    if not points:
        return 0.0
    vals = []
    for p in points:
        if value_key not in p:
            continue
        try:
            vals.append(float(p[value_key]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else 0.0


def normalize_01(values: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    denom = max(vmax - vmin, eps)
    out = (values - vmin) / denom
    return np.clip(out, 0.0, 1.0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
