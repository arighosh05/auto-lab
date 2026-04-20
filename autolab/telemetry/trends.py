"""compute_trends() — windowed slope/mean/std helpers for metric buffers."""

import math
from collections import deque
from typing import Sequence


def _slope(values: Sequence[float]) -> float:
    """Linear regression slope via least squares, using index positions as x.

    Returns 0.0 if fewer than 2 values or if x-variance is zero.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0.0 else 0.0


def _safe_std(values: Sequence[float], mean: float) -> float:
    """Population std; returns 0.0 for single-element sequences."""
    n = len(values)
    if n < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def compute_trends(
    metric_buffer: dict[str, deque],
    window: int = 50,
) -> dict[str, dict[str, float]]:
    """Compute windowed slope, mean, and std for each metric series.

    Args:
        metric_buffer: Maps metric key → deque of recent float values.
            The deque may have any length; we take the last `window` values.
        window: Number of trailing values to include. Default 50 matches
            the default observation cadence — "50-step trends".

    Returns:
        Dict mapping each metric key to {"slope": float, "mean": float, "std": float}.
        Keys with no values produce {"slope": 0.0, "mean": 0.0, "std": 0.0}.
        NaN/Inf values in the buffer are included as-is; caller detects anomalies.
    """
    trends: dict[str, dict[str, float]] = {}
    for key, buf in metric_buffer.items():
        recent = list(buf)[-window:] if buf else []
        if not recent:
            trends[key] = {"slope": 0.0, "mean": 0.0, "std": 0.0}
            continue
        mean = sum(recent) / len(recent)
        trends[key] = {
            "slope": _slope(recent),
            "mean": mean,
            "std": _safe_std(recent, mean),
        }
    return trends
