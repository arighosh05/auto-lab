"""ObservationEvent — the core data structure for telemetry events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ObservationEvent:
    """Snapshot of training state at a given step.

    Emitted by TelemetryLayer every `observation_cadence` training steps
    (default: 50) or immediately when an anomaly is detected.

    Attributes:
        run_id: Training run identifier.
        parent_run_id: run_id of the parent run; None for root runs.
        step: Global training step at time of observation.
        timestamp: Unix timestamp when the event was created.
        metrics: Raw scalar metrics at this step (loss, reward, etc.).
        trends: Windowed statistics over recent steps. Keys mirror metrics
            keys; values are dicts with "slope", "mean", "std".
        anomalies: List of anomaly descriptions when is_anomaly is True.
        config_id: SHA-256[:16] of the sorted config JSON for this run.
        is_anomaly: True if any metric value is NaN or Inf.
    """

    run_id: str
    step: int
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)
    trends: dict[str, dict[str, float]] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    config_id: str = ""
    is_anomaly: bool = False
    parent_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to wire format for JSONL emission.

        Timestamp is emitted as ISO 8601 string (LLM-readable).
        """
        ts_iso = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "step": self.step,
            "timestamp": ts_iso,
            "config_id": self.config_id,
            "metrics": self.metrics,
            "trends": self.trends,
            "anomalies": self.anomalies,
            "is_anomaly": self.is_anomaly,
        }
