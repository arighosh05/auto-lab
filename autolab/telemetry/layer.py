"""TelemetryLayer — drains metrics/sample queues and emits ObservationEvents."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from autolab.telemetry.schema import ObservationEvent
from autolab.telemetry.trends import compute_trends

if TYPE_CHECKING:
    from autolab.store.logs import Logs
    from autolab.store.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

_BUFFER_MAXLEN = 200  # Rolling deque capacity per metric key

# Cadence bounds — prevent the agent from accidentally blinding itself
_OBS_CADENCE_MIN = 10
_OBS_CADENCE_MAX = 200
_SAMPLE_CADENCE_MIN = 5
_SAMPLE_CADENCE_MAX = 100


class TelemetryLayer:
    """Consumes metric and sample entries from queues; emits ObservationEvents.

    Architecture (single-process):
        - start() launches a daemon thread that loops: drain → sleep(0.1)
        - stop() signals the thread and does a final drain before exiting
        - Main thread calls pool.start_run() which blocks in trainer.train()
        - When train() returns, main thread calls stop() then joins the thread

    Cadence control:
        observation_cadence: Emit an ObservationEvent every N training steps.
            Default 50. Agent can adjust per-run via set_cadence().
            Bounds: [10, 200].
        sample_cadence: Store one sample per N accuracy_reward calls.
            Default 25. Agent can adjust per-run via set_cadence().
            Bounds: [5, 100].

    Args:
        metrics_queue: Populated by MetricsHookCallback during training.
        logs: Logs instance for JSONL persistence and pub/sub.
        metadata_store: MetadataStore instance for SQLite persistence.
        observation_cadence: Default steps between ObservationEvent emissions.
        sample_cadence: Default accuracy_reward calls between sample captures.
        sample_queue: Optional queue populated by reward function wrappers.
            Each entry: {"run_id", "reward_fn", "prompt", "completion", "reward"}.
    """

    def __init__(
        self,
        metrics_queue: queue.Queue,
        logs: "Logs",
        metadata_store: "MetadataStore",
        observation_cadence: int = 50,
        sample_cadence: int = 25,
        sample_queue: Optional[queue.Queue] = None,
    ) -> None:
        self._metrics_queue = metrics_queue
        self._sample_queue = sample_queue
        self._logs = logs
        self._store = metadata_store
        self._default_obs_cadence = observation_cadence
        self._default_sample_cadence = sample_cadence

        # Per-run rolling buffers: run_id → metric_key → deque[float]
        self._buffers: dict[str, dict[str, deque]] = {}

        # Per-run cadences (set in register_run, adjustable via set_cadence)
        self._observation_cadences: dict[str, int] = {}
        self._sample_cadences: dict[str, int] = {}

        # Per-run sample call counters (count accuracy_reward calls)
        self._sample_counts: dict[str, int] = {}

        # Tracks training step of last emitted observation per run
        # Initialised to -cadence so first emission fires at or near step 0.
        self._last_emit_step: dict[str, int] = {}

        # Most recently seen step per run (used to tag samples with approximate step)
        self._latest_step: dict[str, int] = {}

        # parent_run_id per run (for ObservationEvent lineage)
        self._parent_run_ids: dict[str, Optional[str]] = {}

        # Config IDs computed in register_run()
        self._config_ids: dict[str, str] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register_run(
        self,
        run_id: str,
        config: dict,
        parent_run_id: Optional[str] = None,
    ) -> None:
        """Register a run and compute its config_id.

        Call this before training starts so config_id is available when
        the first metric event arrives.

        Args:
            run_id: Run identifier.
            config: Full parsed YAML config dict.
            parent_run_id: Parent run's ID for forked runs; None for root runs.
        """
        config_json = json.dumps(config, sort_keys=True, default=str)
        config_id = hashlib.sha256(config_json.encode()).hexdigest()[:16]
        self._config_ids[run_id] = config_id
        self._parent_run_ids[run_id] = parent_run_id
        self._observation_cadences[run_id] = self._default_obs_cadence
        self._sample_cadences[run_id] = self._default_sample_cadence
        self._sample_counts[run_id] = 0
        # Initialise to -cadence: first emission fires at step 0 or the first
        # logged step, giving early visibility into a new run.
        self._last_emit_step[run_id] = -self._default_obs_cadence
        self._latest_step[run_id] = 0
        logger.debug("TelemetryLayer: registered run %s (config_id=%s)", run_id, config_id)

    def set_cadence(
        self,
        run_id: str,
        observation_cadence: Optional[int] = None,
        sample_cadence: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> dict[str, int]:
        """Update per-run emission cadences. Takes effect within 0.1s.

        Values are clamped to valid bounds — the applied values are always
        returned so the caller knows exactly what was set.

        Args:
            run_id: Target run identifier.
            observation_cadence: Steps between ObservationEvent emissions.
                Bounds: [10, 200]. None leaves unchanged.
            sample_cadence: accuracy_reward calls between sample captures.
                Bounds: [5, 100]. None leaves unchanged.
            reason: Optional rationale (not persisted — record in agent trace).

        Returns:
            dict with keys "observation_cadence" and "sample_cadence" showing
            the actual applied values after clamping.
        """
        if observation_cadence is not None:
            applied_obs = max(_OBS_CADENCE_MIN, min(_OBS_CADENCE_MAX, observation_cadence))
            self._observation_cadences[run_id] = applied_obs
        else:
            applied_obs = self._observation_cadences.get(run_id, self._default_obs_cadence)

        if sample_cadence is not None:
            applied_samp = max(_SAMPLE_CADENCE_MIN, min(_SAMPLE_CADENCE_MAX, sample_cadence))
            self._sample_cadences[run_id] = applied_samp
        else:
            applied_samp = self._sample_cadences.get(run_id, self._default_sample_cadence)

        logger.info(
            "TelemetryLayer: set_cadence run=%s obs=%d sample=%d%s",
            run_id, applied_obs, applied_samp,
            f" reason={reason!r}" if reason else "",
        )
        return {"observation_cadence": applied_obs, "sample_cadence": applied_samp}

    def _get_buffer(self, run_id: str, key: str) -> deque:
        """Lazily initialise per-run, per-key rolling buffer."""
        if run_id not in self._buffers:
            self._buffers[run_id] = {}
        if key not in self._buffers[run_id]:
            self._buffers[run_id][key] = deque(maxlen=_BUFFER_MAXLEN)
        return self._buffers[run_id][key]

    def _detect_anomalies(self, metrics: dict[str, float]) -> list[str]:
        """Return a list of anomaly descriptions for NaN or Inf values."""
        anomalies = []
        for key, val in metrics.items():
            if math.isnan(val):
                anomalies.append(f"{key} is NaN")
            elif math.isinf(val):
                sign = "+" if val > 0 else "-"
                anomalies.append(f"{key} is {sign}Inf")
        return anomalies

    def _emit_observation(self, run_id: str, step: int, metrics: dict[str, float]) -> None:
        """Build and dispatch an ObservationEvent to Logs and MetadataStore."""
        buffers = self._buffers.get(run_id, {})
        trends = compute_trends(buffers)
        anomalies = self._detect_anomalies(metrics)

        event = ObservationEvent(
            run_id=run_id,
            parent_run_id=self._parent_run_ids.get(run_id),
            step=step,
            metrics=metrics,
            trends=trends,
            anomalies=anomalies,
            config_id=self._config_ids.get(run_id, ""),
            is_anomaly=bool(anomalies),
        )

        try:
            self._logs.append(run_id, event)
        except Exception:
            logger.exception("TelemetryLayer: failed to write event to Logs")

        try:
            self._store.insert_observation(event)
        except Exception:
            logger.exception("TelemetryLayer: failed to write event to MetadataStore")

        if event.is_anomaly:
            logger.warning(
                "[%s] step=%d  ANOMALY: %s", run_id, step, anomalies
            )

        self._last_emit_step[run_id] = step

    def _drain_metrics(self) -> None:
        """Drain the metrics queue and emit observations as needed."""
        while True:
            try:
                entry = self._metrics_queue.get_nowait()
            except queue.Empty:
                break

            run_id = entry["run_id"]
            step = entry["step"]
            metrics = entry["metrics"]

            # Update rolling buffers
            for key, val in metrics.items():
                self._get_buffer(run_id, key).append(val)

            self._latest_step[run_id] = step

            # Ensure run is registered (fallback if register_run() wasn't called)
            if run_id not in self._last_emit_step:
                self._last_emit_step[run_id] = -self._default_obs_cadence
                self._observation_cadences[run_id] = self._default_obs_cadence

            cadence = self._observation_cadences.get(run_id, self._default_obs_cadence)
            anomalies = self._detect_anomalies(metrics)
            steps_since_emit = step - self._last_emit_step[run_id]
            should_emit = bool(anomalies) or steps_since_emit >= cadence

            if should_emit:
                self._emit_observation(run_id, step, metrics)

            self._metrics_queue.task_done()

    def _drain_samples(self) -> None:
        """Drain the sample queue, sub-sampling by sample_cadence."""
        if self._sample_queue is None:
            return
        while True:
            try:
                entry = self._sample_queue.get_nowait()
            except queue.Empty:
                break

            # Only persist accuracy_reward samples (correctness signal).
            # Each call delivers one completion (completions[0] from the batch).
            if entry.get("reward_fn") == "accuracy_reward":
                run_id = entry["run_id"]
                self._sample_counts[run_id] = self._sample_counts.get(run_id, 0) + 1
                cadence = self._sample_cadences.get(run_id, self._default_sample_cadence)
                if self._sample_counts[run_id] % cadence == 0:
                    step = self._latest_step.get(run_id, 0)
                    try:
                        self._store.insert_sample(
                            run_id=run_id,
                            step=step,
                            prompt=entry["prompt"],
                            completion=entry["completion"],
                            reward=entry["reward"],
                        )
                    except Exception:
                        logger.exception("TelemetryLayer: failed to write sample to MetadataStore")

            self._sample_queue.task_done()

    def process_queue(self) -> None:
        """Drain both queues. Safe to call from any thread."""
        self._drain_metrics()
        self._drain_samples()

    def _drain_loop(self) -> None:
        """Background thread: drain → sleep(0.1) until stop signal."""
        while not self._stop_event.is_set():
            self.process_queue()
            time.sleep(0.1)
        # Final drain after stop is signalled
        self.process_queue()
        logger.info("TelemetryLayer: drain loop exited")

    def start(self) -> None:
        """Start the background drain thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="telemetry-drain",
            daemon=True,
        )
        self._thread.start()
        logger.info("TelemetryLayer: started drain thread")

    def unregister_run(self, run_id: str) -> None:
        """Remove all per-run state for a run that failed to start.

        Called by the control plane during start_run / fork failure rollback so
        telemetry state stays consistent with the store.
        """
        for d in (
            self._buffers,
            self._observation_cadences,
            self._sample_cadences,
            self._sample_counts,
            self._last_emit_step,
            self._latest_step,
            self._parent_run_ids,
            self._config_ids,
        ):
            d.pop(run_id, None)
        logger.debug("TelemetryLayer: unregistered run %s", run_id)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the drain thread to stop and wait for it to exit.

        Args:
            timeout: Seconds to wait for thread join before warning.
        """
        if self._thread is None:
            return
        logger.info("TelemetryLayer: stopping drain thread...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "TelemetryLayer: drain thread did not exit within %ss", timeout
            )
        self._thread = None
