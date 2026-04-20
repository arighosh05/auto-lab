"""SleepScheduler — interruptible sleep for the agent loop.

Subscribes to the telemetry Logs for anomaly events. Uses a threading.Event
polling loop (no asyncio) consistent with the rest of the codebase.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autolab.agent.human_queue import HumanMessageQueue
    from autolab.control_plane.plane import ControlPlane
    from autolab.store.logs import Logs
    from autolab.telemetry.schema import ObservationEvent

_POLL_INTERVAL = 5.0  # seconds between interrupt-source checks


class SleepScheduler:
    """Handles the ``sleep`` tool's wait semantics with four interrupt sources.

    Interrupt sources (first to fire wins):
    1. Timer — natural expiry.
    2. Anomaly — any ObservationEvent with is_anomaly=True received via Logs pub/sub.
    3. Eval complete — any eval flips from 'running' to 'done'/'failed' since sleep started.
    4. Human message — HumanMessageQueue becomes non-empty.

    Additionally, a stop_event threading.Event (set by the .stop file watcher) causes
    woken_by='interrupted'.

    Args:
        logs: Logs instance to subscribe for anomaly events.
        cp: ControlPlane instance; used via get_pending_evals_snapshot() only.
    """

    def __init__(self, logs: "Logs", cp: "ControlPlane") -> None:
        self._anomaly_event = threading.Event()
        self._logs = logs
        self._cp = cp
        logs.subscribe(self._on_observation)

    def _on_observation(self, event: "ObservationEvent") -> None:
        if event.is_anomaly:
            self._anomaly_event.set()

    def wait(
        self,
        seconds: float,
        human_queue: "HumanMessageQueue",
        stop_event: threading.Event,
    ) -> tuple[str, float]:
        """Sleep for up to ``seconds``, returning early on any interrupt.

        Args:
            seconds: Maximum seconds to wait.
            human_queue: Checked for new messages every poll interval.
            stop_event: When set, immediately returns 'interrupted'.

        Returns:
            (woken_by, actual_seconds) where woken_by is one of:
            'timer', 'anomaly', 'eval_complete', 'human', 'interrupted'.
        """
        # Snapshot already-completed evals so we only wake on NEW completions.
        initial_done = {
            eid
            for eid, status in self._cp.get_pending_evals_snapshot().items()
            if status != "running"
        }
        self._anomaly_event.clear()

        deadline = time.monotonic() + seconds
        start = time.monotonic()

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            wait_time = min(_POLL_INTERVAL, max(0.0, remaining))

            # Wait on anomaly event (or timeout)
            if self._anomaly_event.wait(timeout=wait_time):
                self._anomaly_event.clear()
                return "anomaly", time.monotonic() - start

            # Check human messages
            human_queue.poll()
            if not human_queue.empty():
                return "human", time.monotonic() - start

            # Check stop file interrupt
            if stop_event.is_set():
                return "interrupted", time.monotonic() - start

            # Check new eval completions
            for eid, status in self._cp.get_pending_evals_snapshot().items():
                if status != "running" and eid not in initial_done:
                    return "eval_complete", time.monotonic() - start

        return "timer", time.monotonic() - start

    def close(self) -> None:
        """Unsubscribe from Logs. Call when the session ends."""
        self._logs.unsubscribe(self._on_observation)
