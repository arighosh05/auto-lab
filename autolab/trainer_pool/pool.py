"""TrainerPool — manages the registry of training runs."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from typing import Optional

from autolab.trainer_pool.runner import ManagedRun, RunStatus

logger = logging.getLogger(__name__)

# How long pause_run / kill_run poll before declaring a timeout.
_PAUSE_POLL_TIMEOUT = 120.0
# How long to wait for the training thread to fully exit after status is set.
_THREAD_JOIN_TIMEOUT = 15.0


class TrainerPool:
    """Registry and factory for ManagedRun instances.

    Supports two modes of operation:
    - **Blocking** (``start_run``): used by ``scripts/run_training.py`` — backwards-compatible.
    - **Async** (``start_run_async``): used by the control plane — returns immediately.

    At most one run is active (training) at any time.  Other runs are paused
    with their checkpoints preserved.  The active run is tracked in
    ``_active_run_id`` and the training thread in ``_training_thread``; both
    are protected by ``_pool_lock``.

    Lock discipline
    ---------------
    Every read *or* write of ``_active_run_id`` and ``_training_thread`` must
    hold ``_pool_lock``.  ``thread.join()`` calls happen *outside* the lock to
    avoid deadlock (the training thread might try to acquire the lock while
    unwinding).  Pattern::

        with self._pool_lock:
            thread = self._training_thread
        if thread:
            thread.join(timeout=...)
        with self._pool_lock:
            self._training_thread = None
            self._active_run_id = None

    Args:
        metrics_queue: Shared queue passed to each ManagedRun.
            Populated by MetricsHookCallback; consumed by TelemetryLayer.
        sample_queue: Shared queue for captured (prompt, completion, reward) tuples.
            Populated by reward function wrappers; consumed by TelemetryLayer.
    """

    def __init__(self, metrics_queue: queue.Queue, sample_queue: queue.Queue) -> None:
        self._metrics_queue = metrics_queue
        self._sample_queue = sample_queue
        self._runs: dict[str, ManagedRun] = {}

        # Protected by _pool_lock:
        self._active_run_id: Optional[str] = None
        self._training_thread: Optional[threading.Thread] = None
        self._pool_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Backwards-compatible blocking start (used by scripts/run_training.py)
    # -------------------------------------------------------------------------

    def start_run(self, config: dict, run_id: Optional[str] = None) -> str:
        """Create and start a ManagedRun. BLOCKS until training completes.

        Args:
            config: Full parsed YAML config dict.
            run_id: Optional run identifier. If None, generates one from the
                run_name in config and current timestamp.

        Returns:
            run_id string.
        """
        run_id = self.start_run_async(config, run_id=run_id)
        with self._pool_lock:
            thread = self._training_thread
        if thread is not None:
            thread.join()
        return run_id

    # -------------------------------------------------------------------------
    # Async start (used by control plane)
    # -------------------------------------------------------------------------

    def start_run_async(self, config: dict, run_id: Optional[str] = None) -> str:
        """Create a ManagedRun and launch training in a background thread.

        Returns immediately with the run_id.  Use ``get_run(run_id).status`` to
        track progress.

        Args:
            config: Full parsed YAML config dict.
            run_id: Optional run identifier. If None, generates one from the
                run_name in config and current timestamp.

        Returns:
            run_id string.
        """
        if run_id is None:
            run_name = config.get("grpo", {}).get("run_name", "run")
            run_id = f"{run_name}-{int(time.time())}"

        run = ManagedRun(
            run_id=run_id,
            config=config,
            metrics_queue=self._metrics_queue,
            sample_queue=self._sample_queue,
        )

        thread = threading.Thread(
            target=run.start,
            name=f"train-{run_id}",
            daemon=True,
        )

        with self._pool_lock:
            self._runs[run_id] = run
            self._active_run_id = run_id
            self._training_thread = thread

        logger.info("TrainerPool: launching async run %s", run_id)
        thread.start()
        return run_id

    # -------------------------------------------------------------------------
    # Pause / resume / kill
    # -------------------------------------------------------------------------

    def pause_run(self, run_id: str, timeout: float = _PAUSE_POLL_TIMEOUT) -> None:
        """Signal the run to checkpoint and stop.  Blocks until fully exited.

        Sets ``run.pause_requested`` which PauseCallback picks up at the next
        ``on_step_end``.  Polls until status reaches a terminal-for-now state,
        then joins the training thread to guarantee the GPU is free.

        Args:
            run_id: Run to pause.
            timeout: Seconds to poll before raising TimeoutError.
        """
        run = self._get_run_or_raise(run_id)
        if run.status in (RunStatus.PAUSED, RunStatus.DONE, RunStatus.FAILED, RunStatus.KILLED):
            return  # already stopped — idempotent

        with self._pool_lock:
            thread = self._training_thread

        run.pause_requested.set()
        logger.info("TrainerPool: pause signalled for run %s", run_id)

        # Poll for status (set inside trainer thread before it fully exits).
        _TERMINAL = (RunStatus.PAUSED, RunStatus.DONE, RunStatus.FAILED, RunStatus.KILLED)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if run.status in _TERMINAL:
                break
            time.sleep(0.2)
        else:
            raise TimeoutError(
                f"pause_run: run {run_id!r} did not reach a stopped state "
                f"within {timeout}s (current status: {run.status.value})"
            )

        # Join the thread *outside* the lock — status is set before train()
        # returns but the thread is still alive until start() fully unwinds.
        # Without join, resume_run could start a new thread before the first
        # fully released the GPU.
        if thread is not None:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning(
                    "TrainerPool: training thread for %s did not exit within %ss after pause",
                    run_id, _THREAD_JOIN_TIMEOUT,
                )

        with self._pool_lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
                self._training_thread = None

        logger.info("TrainerPool: run %s paused (status=%s)", run_id, run.status.value)

    def resume_run(self, run_id: str) -> None:
        """Re-launch a PAUSED run from its last saved checkpoint.

        Joins the previous training thread (if any) before starting the new one
        so no two threads ever touch the GPU simultaneously.

        Args:
            run_id: Run to resume (must be in PAUSED status).
        """
        run = self._get_run_or_raise(run_id)
        if run.status != RunStatus.PAUSED:
            raise ValueError(
                f"resume_run: run {run_id!r} is {run.status.value!r}, not 'paused'"
            )

        # Snapshot and join old thread outside the lock.
        with self._pool_lock:
            old_thread = self._training_thread
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=_THREAD_JOIN_TIMEOUT)

        # Inject resume_from_checkpoint so TRL restores weights + optimizer.
        config = copy.deepcopy(run.config)
        config.setdefault("grpo", {})["resume_from_checkpoint"] = run.checkpoint_path

        # Reset control signals before re-entering train().
        run.pause_requested.clear()
        run.kill_requested.clear()
        run.config = config

        thread = threading.Thread(
            target=run.start,
            name=f"train-{run_id}",
            daemon=True,
        )

        with self._pool_lock:
            self._active_run_id = run_id
            self._training_thread = thread

        logger.info(
            "TrainerPool: resuming run %s from checkpoint %s",
            run_id, run.checkpoint_path,
        )
        thread.start()

    def kill_run(self, run_id: str, timeout: float = _PAUSE_POLL_TIMEOUT) -> None:
        """Signal the run to checkpoint and stop permanently.

        Idempotent: calling on an already-killed / done / failed run is a no-op.

        Args:
            run_id: Run to kill.
            timeout: Seconds to poll before giving up on graceful stop.
        """
        run = self._get_run_or_raise(run_id)
        if run.status in (RunStatus.KILLED, RunStatus.DONE, RunStatus.FAILED):
            return  # already terminal — idempotent

        with self._pool_lock:
            thread = self._training_thread

        # Both events set: PauseCallback checks either one.
        run.kill_requested.set()
        run.pause_requested.set()
        logger.info("TrainerPool: kill signalled for run %s", run_id)

        # If the run is already stopped (PAUSED), the training thread is gone and
        # cannot set status to KILLED itself.  Do it directly.
        if run.status == RunStatus.PAUSED:
            run.status = RunStatus.KILLED

        _TERMINAL = (RunStatus.KILLED, RunStatus.DONE, RunStatus.FAILED, RunStatus.PAUSED)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if run.status in _TERMINAL:
                break
            time.sleep(0.2)
        else:
            logger.warning(
                "TrainerPool: run %s did not stop within %ss — thread may still be alive",
                run_id, timeout,
            )

        if thread is not None:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.warning(
                    "TrainerPool: training thread for %s still alive after kill join",
                    run_id,
                )

        with self._pool_lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
                self._training_thread = None

        logger.info("TrainerPool: run %s killed (status=%s)", run_id, run.status.value)

    # -------------------------------------------------------------------------
    # Hot-modify (applied at next on_step_begin by HotModifyCallback)
    # -------------------------------------------------------------------------

    def modify_run(self, run_id: str, overrides: dict) -> None:
        """Push overrides into pending_mods; applied within one training step.

        Args:
            run_id: Target run (running or paused).
            overrides: {param: new_value} — must be in the allowlist.
        """
        run = self._get_run_or_raise(run_id)
        run.pending_mods.update(overrides)
        logger.debug(
            "TrainerPool: queued modifications for run %s: %s", run_id, list(overrides)
        )

    # -------------------------------------------------------------------------
    # Fork helper (control plane does the checkpoint copy and store writes)
    # -------------------------------------------------------------------------

    def fork_run(
        self,
        child_run_id: str,
        child_config: dict,
    ) -> None:
        """Start a child run from a pre-prepared config (checkpoint path already set).

        The control plane handles copying the parent checkpoint, computing the
        new config_id, and registering the run in the store *before* calling
        this method.  This method only creates the ManagedRun and launches its
        thread.

        Args:
            child_run_id: Identifier for the new run.
            child_config: Full config dict with ``grpo.resume_from_checkpoint``
                already pointing at the copied checkpoint.
        """
        run = ManagedRun(
            run_id=child_run_id,
            config=child_config,
            metrics_queue=self._metrics_queue,
            sample_queue=self._sample_queue,
        )
        thread = threading.Thread(
            target=run.start,
            name=f"train-{child_run_id}",
            daemon=True,
        )

        with self._pool_lock:
            self._runs[child_run_id] = run
            self._active_run_id = child_run_id
            self._training_thread = thread

        logger.info("TrainerPool: launching forked run %s", child_run_id)
        thread.start()

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_active_run_id(self) -> Optional[str]:
        """Return the run_id of the currently training run, or None."""
        with self._pool_lock:
            return self._active_run_id

    def get_paused_run_ids(self) -> list[str]:
        """Return run_ids of all paused (resumable) runs."""
        return [rid for rid, r in self._runs.items() if r.status == RunStatus.PAUSED]

    def list_runs(self) -> list[dict]:
        """Return status dicts for all runs."""
        return [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "current_step": r.current_step,
            }
            for r in self._runs.values()
        ]

    def get_run(self, run_id: str) -> Optional[ManagedRun]:
        """Retrieve a ManagedRun by ID, or None if not found."""
        return self._runs.get(run_id)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_run_or_raise(self, run_id: str) -> ManagedRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"TrainerPool: run {run_id!r} not found")
        return run
