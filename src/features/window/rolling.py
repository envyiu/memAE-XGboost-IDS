"""Low-level rolling window primitives and entropy helpers."""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

# --- Entropy cache (lazy-loaded) ---

MAX_Q = 200000
_F_LOG_F_CACHE: np.ndarray | None = None


def get_f_log_f() -> np.ndarray:
    """Lazy-init and return the f*log2(f) lookup table."""
    global _F_LOG_F_CACHE
    if _F_LOG_F_CACHE is None:
        _F_LOG_F_CACHE = np.zeros(MAX_Q, dtype=np.float32)
        for i in range(1, MAX_Q):
            _F_LOG_F_CACHE[i] = i * math.log2(i)
    return _F_LOG_F_CACHE


def entropy(count: float, sum_f_log_f: float) -> float:
    """Compute Shannon entropy from running f*log(f) accumulator."""
    if count <= 1:
        return 0.0
    return max(0.0, math.log2(count) - sum_f_log_f / count)


# --- Numpy rolling helpers ---

def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum via cumsum trick."""
    if values.size == 0:
        return values.astype("float32")
    cumsum = np.cumsum(values, dtype=np.float64)
    out = cumsum.copy()
    if window < len(values):
        out[window:] = cumsum[window:] - cumsum[:-window]
    return out.astype("float32")


def rolling_binary_ratio(values: np.ndarray, window: int, flow_count: np.ndarray) -> np.ndarray:
    """Rolling ratio of binary indicator over flow count."""
    summed = rolling_sum(values.astype(np.float32), window)
    return summed / np.maximum(flow_count, 1.0)


# --- Count-based sliding window helpers ---

def window_port_stats(ports: np.ndarray, window_sizes: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute per-port hit count and unique port count within each count-based window."""
    unique_counts: dict[int, np.ndarray] = {}
    port_counts: dict[int, np.ndarray] = {}
    for window in window_sizes:
        counts = np.zeros(len(ports), dtype=np.int32)
        uniques = np.zeros(len(ports), dtype=np.int32)
        queue: deque[int] = deque()
        counter: dict[int, int] = {}
        for i, port in enumerate(ports.tolist()):
            queue.append(port)
            counter[port] = counter.get(port, 0) + 1
            if len(queue) > window:
                old = queue.popleft()
                counter[old] -= 1
                if counter[old] <= 0:
                    counter.pop(old, None)
            counts[i] = counter.get(port, 0)
            uniques[i] = len(counter)
        port_counts[window] = counts
        unique_counts[window] = uniques
    return port_counts, unique_counts


def window_value_counts(values: np.ndarray, window_sizes: list[int]) -> dict[int, np.ndarray]:
    """Count occurrences of the current value within each count-based window."""
    value_counts: dict[int, np.ndarray] = {}
    for window in window_sizes:
        counts = np.zeros(len(values), dtype=np.int32)
        queue: deque[Any] = deque()
        counter: dict[Any, int] = {}
        for i, value in enumerate(values.tolist()):
            queue.append(value)
            counter[value] = counter.get(value, 0) + 1
            if len(queue) > window:
                old = queue.popleft()
                counter[old] -= 1
                if counter[old] <= 0:
                    counter.pop(old, None)
            counts[i] = counter.get(value, 0)
        value_counts[window] = counts
    return value_counts


def to_unix_seconds(values: "pd.Series") -> np.ndarray:
    """Convert a timestamp series to int64 Unix seconds, with fallbacks."""
    import pandas as pd

    dt = values
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", dayfirst=True)
        if pd.isna(dt).all():
            dt = pd.to_datetime(values, errors="coerce", dayfirst=False)
    if isinstance(dt, pd.Series):
        dt_series = dt
    else:
        dt_series = pd.Series(dt)
    if dt_series.isna().any():
        dt_series = dt_series.ffill().bfill()
    if dt_series.isna().any():
        fallback = pd.Series(np.arange(len(dt_series), dtype=np.int64), index=dt_series.index)
        dt_series = dt_series.where(~dt_series.isna(), pd.to_datetime(fallback, unit="s"))
    return (dt_series.astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)
