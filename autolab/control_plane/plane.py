"""ControlPlane — the agent-facing API for the autolab training system.

14 tools (7 write, 7 read) that sit above the trainer pool, telemetry layer,
and store.  The agent calls these; nothing below is visible to the agent.

Write tools
-----------
start_run, fork, kill, modify, set_active, set_cadence, eval

Read tools
----------
get_run_details, list_runs, get_history, get_sample, compute_trend,
get_config, get_eval
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from autolab.control_plane.types import ErrorCode, StatusBar, ToolError, ToolResponse
from autolab.trainer_pool.runner import RunStatus

if TYPE_CHECKING:
    from autolab.store.logs import Logs
    from autolab.store.metadata_store import MetadataStore
    from autolab.telemetry.layer import TelemetryLayer
    from autolab.trainer_pool.pool import TrainerPool

logger = logging.getLogger(__name__)

# Hot-modifiable parameter allowlist (must match pause_callback._HOT_MODIFY_ALLOWLIST).
_MODIFY_ALLOWLIST = frozenset(
    {"learning_rate", "beta", "epsilon", "temperature", "top_p", "top_k"}
)

# Supported benchmark names for eval().
_KNOWN_BENCHMARKS = {"math"}

# Required top-level config keys for start_run / fork.
_REQUIRED_CONFIG_KEYS = {"model_name", "dataset_name", "grpo"}


class ControlPlane:
    """Agent-facing API.  Wraps TrainerPool, MetadataStore, and TelemetryLayer.

    Args:
        pool: TrainerPool instance (owns ManagedRun lifecycle).
        store: MetadataStore instance (SQLite persistence).
        telemetry: TelemetryLayer instance (metrics drain + cadence control).
        logs: Logs instance (JSONL event stream).
    """

    def __init__(
        self,
        pool: "TrainerPool",
        store: "MetadataStore",
        telemetry: "TelemetryLayer",
        logs: "Logs",
    ) -> None:
        self._pool = pool
        self._store = store
        self._telemetry = telemetry
        self._logs = logs

        # eval_id → {"status", "run_id", "paused_run_id", "results", "error", "thread"}
        self._pending_evals: dict[str, dict] = {}
        # Lock protecting _pending_evals mutations from eval threads.
        self._eval_lock = threading.Lock()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _status(self) -> StatusBar:
        with self._eval_lock:
            pending = [eid for eid, e in self._pending_evals.items()
                       if e["status"] == "running"]
            evaluating = bool(pending)
        active_id = self._pool.get_active_run_id()
        return StatusBar(
            active_run_id=active_id,
            paused_run_ids=self._pool.get_paused_run_ids(),
            pending_evals=pending,
            gpu_state=(
                "evaluating" if evaluating
                else "training" if active_id
                else "idle"
            ),
        )

    def _ok(self, result: Any) -> ToolResponse:
        return ToolResponse(status=self._status(), result=result)

    def _err(
        self,
        code: str,
        message: str,
        retryable: bool,
        suggested_action: str,
    ) -> ToolResponse:
        return ToolResponse(
            status=self._status(),
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                suggested_action=suggested_action,
            ),
        )

    def _config_id(self, config: dict) -> str:
        s = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _validate_config(self, config: dict) -> Optional[str]:
        """Return an error message if config is missing required keys, else None."""
        missing = _REQUIRED_CONFIG_KEYS - set(config)
        if missing:
            return f"Config missing required keys: {sorted(missing)}"
        return None

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """Glob {output_dir}/checkpoint-N and return the highest-numbered path.

        Only considers checkpoints with a purely numeric suffix (checkpoint-5,
        checkpoint-10, …).  Non-standard names like checkpoint-fork-21 are
        excluded — they may be deleted by TRL's rotation logic even though they
        don't appear in its sorted list, making them unreliable for eval.
        """
        import re as _re
        p = Path(output_dir)
        candidates = [
            c for c in p.glob("checkpoint-*")
            if c.is_dir() and _re.fullmatch(r"checkpoint-\d+", c.name)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: int(c.name.split("-")[-1]))
        return str(candidates[-1])

    # =========================================================================
    # ── Write tools ──────────────────────────────────────────────────────────
    # =========================================================================

    def start_run(self, config: dict, reason: str) -> ToolResponse:
        """Create a new training run from scratch.

        Args:
            config: Full training config dict (model_name, dataset_name, grpo, …).
            reason: Why this run was created.  Written to the store audit trail.

        Returns:
            ToolResponse with result ``{"run_id": str, "config_id": str}``.
        """
        err = self._validate_config(config)
        if err:
            return self._err(ErrorCode.INVALID_CONFIG, err, False,
                             "Provide a config with at least model_name, dataset_name, and grpo keys.")

        run_name = config.get("grpo", {}).get("run_name", "run")
        run_id = f"{run_name}-{int(time.time())}"
        config_id = self._config_id(config)

        # Persist config + run *before* launching the thread so the store is
        # consistent even if the trainer fails immediately.
        self._store.insert_config(config_id, json.dumps(config, default=str))
        self._store.insert_run(
            run_id=run_id,
            config_id=config_id,
            status="running",
            start_time=time.time(),
            creation_reason=reason,
        )
        self._telemetry.register_run(run_id, config)

        try:
            self._pool.start_run_async(config, run_id=run_id)
        except Exception as exc:
            # Rollback: mark run failed; unregister from telemetry.
            self._store.update_run(run_id, "failed", time.time(), "{}")
            self._telemetry.unregister_run(run_id)
            return self._err(
                ErrorCode.TRAINER_START_FAILED,
                f"Trainer failed to launch: {exc}",
                True,
                "Check model_name and dataset_name are accessible, then retry.",
            )

        logger.info("ControlPlane: started run %s (config_id=%s)", run_id, config_id)
        return self._ok({"run_id": run_id, "config_id": config_id})

    def fork(
        self,
        parent_run_id: str,
        overrides: dict,
        reason: str,
    ) -> ToolResponse:
        """Create a child run from the parent's latest checkpoint with config overrides.

        The parent is paused (if currently running), its checkpoint is copied to
        a new location, and the child starts training from that copy.  The parent
        remains paused — use ``set_active(parent_run_id)`` to resume it later.

        Args:
            parent_run_id: Run to fork from.
            overrides: Config keys/values to change relative to the parent config.
                Top-level keys or ``grpo.*`` subkeys are both accepted.
            reason: Why this fork was created.

        Returns:
            ToolResponse with result ``{"run_id", "parent_run_id", "fork_step", "config_id"}``.
        """
        parent_run = self._pool.get_run(parent_run_id)
        if parent_run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {parent_run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )
        if parent_run.status in (RunStatus.FAILED, RunStatus.KILLED):
            return self._err(
                ErrorCode.RUN_NOT_FORKABLE,
                f"Cannot fork run {parent_run_id!r} because its status is "
                f"{parent_run.status.value!r}.",
                False,
                f"Use start_run() with an explicit config instead. "
                f"Get the parent config with get_config('{self._telemetry._config_ids.get(parent_run_id, '')}').",
            )

        # ── Step 1: pause parent if currently running ──────────────────────
        was_running = (parent_run.status == RunStatus.RUNNING)
        if was_running:
            try:
                self._pool.pause_run(parent_run_id)
            except TimeoutError as exc:
                return self._err(
                    ErrorCode.PAUSE_TIMEOUT,
                    str(exc),
                    True,
                    "Retry the fork — the trainer may have been mid-step.",
                )

        fork_step = parent_run.current_step

        # ── Step 2: locate checkpoint ──────────────────────────────────────
        parent_checkpoint = parent_run.checkpoint_path
        parent_output_dir = parent_run.config.get("grpo", {}).get("output_dir", "")
        if not parent_checkpoint and parent_output_dir:
            parent_checkpoint = self._find_latest_checkpoint(parent_output_dir)
        if not parent_checkpoint:
            if was_running:
                self._pool.resume_run(parent_run_id)
            return self._err(
                ErrorCode.NO_CHECKPOINT,
                f"Run {parent_run_id!r} has no checkpoint to fork from.",
                False,
                "Ensure the run has trained past the first save_steps before forking.",
            )

        # ── Step 3: build child config ─────────────────────────────────────
        child_config = copy.deepcopy(parent_run.config)
        for k, v in overrides.items():
            if "." in k:
                section, subkey = k.split(".", 1)
                child_config.setdefault(section, {})[subkey] = v
            else:
                child_config[k] = v

        child_run_id = f"{child_config.get('grpo', {}).get('run_name', 'run')}-fork-{int(time.time())}"
        child_output_dir = f"outputs/{child_run_id}"
        child_checkpoint_dest = str(Path(child_output_dir) / f"checkpoint-fork-{fork_step}")

        child_config.setdefault("grpo", {})["output_dir"] = child_output_dir
        child_config["grpo"]["resume_from_checkpoint"] = child_checkpoint_dest

        child_config_id = self._config_id(child_config)

        # ── Step 4: copy checkpoint ────────────────────────────────────────
        try:
            shutil.copytree(parent_checkpoint, child_checkpoint_dest)
        except Exception as exc:
            if was_running:
                try:
                    self._pool.resume_run(parent_run_id)
                except Exception:
                    logger.exception("fork rollback: failed to resume parent %s", parent_run_id)
            return self._err(
                ErrorCode.CHECKPOINT_COPY_FAILED,
                f"Failed to copy checkpoint: {exc}",
                True,
                "Check disk space and retry.",
            )

        # ── Step 5: persist to store ───────────────────────────────────────
        self._store.insert_config(child_config_id, json.dumps(child_config, default=str))
        self._store.insert_run(
            run_id=child_run_id,
            config_id=child_config_id,
            status="running",
            start_time=time.time(),
            parent_run_id=parent_run_id,
            fork_step=fork_step,
            creation_reason=reason,
        )
        self._telemetry.register_run(child_run_id, child_config, parent_run_id=parent_run_id)

        # ── Step 6: launch child training thread ───────────────────────────
        try:
            self._pool.fork_run(child_run_id=child_run_id, child_config=child_config)
        except Exception as exc:
            # Rollback store + telemetry + checkpoint copy
            self._store.update_run(child_run_id, "failed", time.time(), "{}")
            self._telemetry.unregister_run(child_run_id)
            try:
                shutil.rmtree(child_checkpoint_dest, ignore_errors=True)
            except Exception:
                pass
            if was_running:
                try:
                    self._pool.resume_run(parent_run_id)
                except Exception:
                    logger.exception("fork rollback: failed to resume parent %s", parent_run_id)
            return self._err(
                ErrorCode.TRAINER_START_FAILED,
                f"Child trainer failed to launch: {exc}",
                True,
                "Retry the fork.",
            )

        logger.info(
            "ControlPlane: forked %s → %s at step %d (config_id=%s)",
            parent_run_id, child_run_id, fork_step, child_config_id,
        )
        return self._ok({
            "run_id": child_run_id,
            "parent_run_id": parent_run_id,
            "fork_step": fork_step,
            "config_id": child_config_id,
        })

    def kill(self, run_id: str, reason: str) -> ToolResponse:
        """Stop a run permanently.  Idempotent.

        If the run is currently active the trainer checkpoints cleanly before
        stopping.  The GPU enters idle state; the agent must call ``set_active``
        to continue work.

        Args:
            run_id: Run to kill.
            reason: Why this run was killed (for agent trace only — not stored).

        Returns:
            ToolResponse with result ``{"run_id": str, "status": "killed"}``.
        """
        run = self._pool.get_run(run_id)
        if run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )
        # Idempotent: already terminal.
        if run.status in (RunStatus.KILLED, RunStatus.DONE, RunStatus.FAILED):
            return self._ok({"run_id": run_id, "status": run.status.value})

        try:
            self._pool.kill_run(run_id)
        except Exception as exc:
            return self._err(
                ErrorCode.PAUSE_TIMEOUT,
                f"Kill signal sent but trainer did not stop: {exc}",
                True,
                "Retry kill — the trainer may be mid-step.",
            )

        self._store.update_run(
            run_id=run_id,
            status="killed",
            end_time=run.end_time or time.time(),
            final_metrics_json=json.dumps(run.final_metrics, default=str) if run.final_metrics else "{}",
        )
        return self._ok({"run_id": run_id, "status": "killed"})

    def modify(
        self,
        run_id: str,
        overrides: dict,
        reason: str,
    ) -> ToolResponse:
        """Hot-modify hyperparameters on a run without restarting it.

        Only allowlisted params can be modified in-flight:
        ``learning_rate``, ``beta``, ``epsilon``, ``temperature``, ``top_p``,
        ``top_k``.  For any other param, use ``fork`` with overrides instead.

        Changes take effect within one training step (applied by HotModifyCallback
        at the next ``on_step_begin``).  Works on both running and paused runs.

        Args:
            run_id: Target run.
            overrides: ``{param: new_value}`` — all keys must be in the allowlist.
            reason: Why this modification is being made.

        Returns:
            ToolResponse with result ``{"run_id", "new_config_id", "applied"}``.
        """
        run = self._pool.get_run(run_id)
        if run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )
        if run.status not in (RunStatus.RUNNING, RunStatus.PAUSED):
            return self._err(
                ErrorCode.RUN_NOT_RESUMABLE,
                f"Run {run_id!r} is {run.status.value!r} — cannot modify.",
                False,
                "Only running or paused runs can be modified.",
            )

        # Validate allowlist.
        bad_params = set(overrides) - _MODIFY_ALLOWLIST
        if bad_params:
            first_bad = sorted(bad_params)[0]
            return self._err(
                ErrorCode.PARAM_NOT_MODIFIABLE,
                f"Parameter(s) {sorted(bad_params)} are not hot-modifiable.",
                False,
                f"Use fork() with overrides={{{first_bad!r}: ...}} to change these "
                f"parameters without restarting training.",
            )

        # Basic range validation.
        for param, value in overrides.items():
            if param == "learning_rate" and not (0 < value):
                return self._err(ErrorCode.PARAM_OUT_OF_RANGE,
                                 f"learning_rate must be > 0, got {value}", False, "")
            if param == "beta" and value < 0:
                return self._err(ErrorCode.PARAM_OUT_OF_RANGE,
                                 f"beta must be >= 0, got {value}", False, "")
            if param == "temperature" and not (value > 0):
                return self._err(ErrorCode.PARAM_OUT_OF_RANGE,
                                 f"temperature must be > 0, got {value}", False, "")

        # Build new config + config_id.
        old_config_id = self._telemetry._config_ids.get(run_id, "")
        new_config = copy.deepcopy(run.config)
        for k, v in overrides.items():
            new_config.setdefault("grpo", {})[k] = v
        new_config_id = self._config_id(new_config)

        # Persist audit trail.
        self._store.insert_config(new_config_id, json.dumps(new_config, default=str))
        step = run.current_step
        self._store.insert_modification(
            run_id=run_id,
            step=step,
            old_config_id=old_config_id,
            new_config_id=new_config_id,
            changes_json=json.dumps(overrides, default=str),
            reason=reason,
        )

        # Push to callback + update telemetry config tracking.
        self._pool.modify_run(run_id, overrides)
        self._telemetry._config_ids[run_id] = new_config_id

        return self._ok({
            "run_id": run_id,
            "new_config_id": new_config_id,
            "applied": list(overrides.keys()),
            "step": step,
        })

    def set_active(self, run_id: Optional[str], reason: str) -> ToolResponse:
        """Change which run is consuming the GPU.

        Pauses whatever is currently active (saving a checkpoint), then resumes
        the named run from its last checkpoint.  Pass ``run_id=None`` to leave
        the GPU idle.  Idempotent: setting active to the already-active run is
        a no-op success.

        Args:
            run_id: Run to make active, or None for idle.
            reason: Why the switch is being made (for agent trace only).

        Returns:
            ToolResponse with result ``{"active_run_id": Optional[str]}``.
        """
        current_active = self._pool.get_active_run_id()

        # Idempotent check.
        if run_id == current_active:
            return self._ok({"active_run_id": run_id})

        # Pause whatever is currently active.
        if current_active is not None:
            try:
                self._pool.pause_run(current_active)
                self._store.update_run(current_active, "paused",
                                       self._pool.get_run(current_active).end_time or time.time(),
                                       "{}")
            except TimeoutError as exc:
                return self._err(
                    ErrorCode.PAUSE_TIMEOUT, str(exc), True,
                    "Retry set_active — the trainer may be mid-step.",
                )

        if run_id is None:
            return self._ok({"active_run_id": None})

        # Validate target run.
        run = self._pool.get_run(run_id)
        if run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )
        if run.status != RunStatus.PAUSED:
            return self._err(
                ErrorCode.RUN_NOT_RESUMABLE,
                f"Run {run_id!r} is {run.status.value!r} — only paused runs can be resumed.",
                False,
                "Use start_run() or fork() to begin a new run.",
            )

        self._pool.resume_run(run_id)
        self._store.update_run(run_id, "running", None, "{}")  # type: ignore[arg-type]

        return self._ok({"active_run_id": run_id})

    def set_cadence(
        self,
        run_id: str,
        observation_cadence: Optional[int] = None,
        sample_cadence: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> ToolResponse:
        """Adjust observation and sample capture frequencies for a run.

        Values are clamped to valid bounds [10, 200] and [5, 100] respectively.
        The actual applied values are always returned so the agent knows exactly
        what was set.  Idempotent.

        Args:
            run_id: Target run.
            observation_cadence: Steps between ObservationEvent emissions.
            sample_cadence: accuracy_reward calls between sample captures.
            reason: Why the cadence is being changed (not persisted).

        Returns:
            ToolResponse with result ``{"observation_cadence": int, "sample_cadence": int}``.
        """
        run = self._pool.get_run(run_id)
        if run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )
        applied = self._telemetry.set_cadence(
            run_id,
            observation_cadence=observation_cadence,
            sample_cadence=sample_cadence,
            reason=reason,
        )
        return self._ok(applied)

    def eval(
        self,
        run_id: str,
        benchmark: str,
        n_samples: int,
        reason: str,
    ) -> ToolResponse:
        """Trigger an async evaluation of a run's latest checkpoint.

        Returns immediately with an eval_id and "running" status.  The active
        training run is paused to free the GPU for inference.  When eval
        finishes, the system enters idle state — the agent must call
        ``set_active`` to resume training.

        Args:
            run_id: Run whose checkpoint to evaluate.
            benchmark: Benchmark name.  Currently only "math" is supported.
            n_samples: Number of test samples (-1 for full test set).
            reason: Why this eval was triggered (for agent trace only).

        Returns:
            ToolResponse with result ``{"eval_id", "status", "estimated_seconds",
            "training_paused"}``.
        """
        if benchmark not in _KNOWN_BENCHMARKS:
            return self._err(
                ErrorCode.BENCHMARK_UNKNOWN,
                f"Benchmark {benchmark!r} not recognised.  Known: {sorted(_KNOWN_BENCHMARKS)}.",
                False,
                "Use 'math' as the benchmark name.",
            )

        run = self._pool.get_run(run_id)
        if run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )

        # Find checkpoint.
        checkpoint_path = run.checkpoint_path
        output_dir = run.config.get("grpo", {}).get("output_dir", "")
        if not checkpoint_path and output_dir:
            checkpoint_path = self._find_latest_checkpoint(output_dir)
        if not checkpoint_path:
            return self._err(
                ErrorCode.NO_CHECKPOINT,
                f"Run {run_id!r} has no saved checkpoint to evaluate.",
                False,
                "Wait for the run to reach the first save_steps, then retry.",
            )

        # Pause active run if any.
        active_id = self._pool.get_active_run_id()
        paused_for_eval: Optional[str] = None
        if active_id is not None:
            try:
                self._pool.pause_run(active_id)
                paused_for_eval = active_id
            except TimeoutError as exc:
                return self._err(
                    ErrorCode.PAUSE_TIMEOUT, str(exc), True,
                    "Retry eval — the trainer may be mid-step.",
                )

        eval_id = f"eval-{run_id[:8]}-{int(time.time())}"
        eval_state: dict = {
            "status": "running",
            "run_id": run_id,
            "paused_run_id": paused_for_eval,
            "results": None,
            "error": None,
        }
        with self._eval_lock:
            self._pending_evals[eval_id] = eval_state

        # Launch eval in background thread.
        step = run.current_step
        thread = threading.Thread(
            target=self._eval_worker,
            args=(eval_id, eval_state, checkpoint_path, run.config, n_samples, run_id, step),
            name=f"eval-{eval_id}",
            daemon=True,
        )
        eval_state["thread"] = thread
        thread.start()

        estimated_seconds = max(60, n_samples * 2)
        return self._ok({
            "eval_id": eval_id,
            "status": "running",
            "estimated_seconds": estimated_seconds,
            "training_paused": paused_for_eval,
        })

    def _eval_worker(
        self,
        eval_id: str,
        eval_state: dict,
        checkpoint_path: str,
        config: dict,
        n_samples: int,
        run_id: str,
        checkpoint_step: int,
    ) -> None:
        """Background thread: run evaluation, write result to store."""
        try:
            from autolab.eval.evaluator import (
                run_eval,  # lazy: avoids torch/datasets at import
            )
            results = run_eval(
                checkpoint_path=checkpoint_path,
                config=config,
                n_samples=n_samples,
            )
            self._store.insert_eval(
                run_id=run_id,
                checkpoint_step=checkpoint_step,
                benchmark="math",
                n_samples=results["n_samples"],
                accuracy=results["overall_accuracy"],
                detailed_results_json=json.dumps(
                    {"by_level": results["by_level"], "by_type": results["by_type"]},
                    default=str,
                ),
            )
            with self._eval_lock:
                eval_state["status"] = "done"
                eval_state["results"] = results
            logger.info(
                "ControlPlane: eval %s complete — accuracy=%.4f",
                eval_id, results["overall_accuracy"],
            )
        except Exception as exc:
            with self._eval_lock:
                eval_state["status"] = "failed"
                eval_state["error"] = str(exc)
            logger.exception("ControlPlane: eval %s failed", eval_id)
        # No auto-resume.  System enters idle state.  Agent calls set_active().

    # =========================================================================
    # ── Read tools ───────────────────────────────────────────────────────────
    # =========================================================================

    def get_run_details(self, run_id: str) -> ToolResponse:
        """Return everything the system knows about a single run.

        Includes: lineage, config, modification history, latest observation,
        post-hoc statistics (AUC, convergence, stability), completed evals.
        Post-hoc stats are computed at query time from stored observations.

        Args:
            run_id: Run identifier.
        """
        run = self._pool.get_run(run_id)
        store_run = self._store.get_run(run_id)
        if store_run is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} not found.",
                False,
                "Call list_runs() to see available run IDs.",
            )

        config_id = store_run.get("config_id", "")
        config = self._store.get_config(config_id) if config_id else None

        history = self._store.get_history(run_id)
        latest_obs = history[-1] if history else None

        modifications = self._store.list_modifications(run_id)
        evals = self._store.list_evals(run_id)

        post_hoc = self._compute_post_hoc_stats(history)

        return self._ok({
            "run": store_run,
            "config": config,
            "latest_observation": latest_obs,
            "modifications": modifications[-10:],  # last 10
            "evals": evals,
            "post_hoc_stats": post_hoc,
            "live": {
                "status": run.status.value if run else store_run.get("status"),
                "current_step": run.current_step if run else None,
                "checkpoint_path": run.checkpoint_path if run else None,
            },
        })

    def _compute_post_hoc_stats(self, history: list[dict]) -> dict:
        """Compute AUC, convergence step, stability, sample efficiency."""
        if len(history) < 3:
            return {"note": "insufficient data (need ≥ 3 observations)", "computed_over_steps": []}

        steps = [h["step"] for h in history]
        metrics_keys = list(history[0]["metrics"].keys())
        stats: dict[str, Any] = {"computed_over_steps": [steps[0], steps[-1]]}

        for key in metrics_keys:
            vals = [h["metrics"].get(key) for h in history]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            # AUC via trapezoid (normalised to step range).
            step_range = steps[-1] - steps[0]
            if step_range > 0:
                auc = sum(
                    (v1 + v2) / 2 * (s2 - s1)
                    for (s1, v1), (s2, v2)
                    in zip(zip(steps, vals), zip(steps[1:], vals[1:]))
                ) / step_range
            else:
                auc = mean
            # Stability: std of last 20% of values.
            tail = max(1, len(vals) // 5)
            stability = math.sqrt(
                sum((v - sum(vals[-tail:]) / tail) ** 2 for v in vals[-tail:]) / tail
            )
            stats[key] = {"mean": mean, "std": std, "auc": auc, "stability": stability}

        return stats

    def list_runs(self, status_filter: Optional[list[str]] = None) -> ToolResponse:
        """List runs with optional status filter.

        Default filter shows only "running" and "paused" runs.  Pass an explicit
        list to include "done", "failed", "killed", "pending".

        Args:
            status_filter: Status values to include, or None for default.
        """
        if status_filter is None:
            status_filter = ["running", "paused"]

        store_runs = self._store.list_runs()
        result = []
        for sr in store_runs:
            if sr["status"] not in status_filter:
                continue
            rid = sr["run_id"]
            pool_run = self._pool.get_run(rid)
            history = self._store.get_history(rid)
            latest_acc = None
            current_step = sr.get("current_step") or 0
            if history:
                latest_acc = history[-1]["metrics"].get("rewards/accuracy_reward/mean")
                current_step = history[-1]["step"]
            if pool_run:
                current_step = pool_run.current_step or current_step
            result.append({
                "run_id": rid,
                "status": sr["status"],
                "parent_run_id": sr.get("parent_run_id"),
                "current_step": current_step,
                "latest_accuracy": latest_acc,
                "creation_reason": sr.get("creation_reason"),
            })
        return self._ok(result)

    def get_history(
        self,
        run_id: str,
        step_range: Optional[tuple[int, int]] = None,
        fields: Optional[list[str]] = None,
    ) -> ToolResponse:
        """Return observation events for a run.

        Args:
            run_id: Run identifier.
            step_range: Optional (min_step, max_step) tuple, inclusive.
            fields: If given, only these metric keys are included in each
                observation's ``metrics`` dict.
        """
        run = self._store.get_run(run_id)
        if run is None:
            return self._err(ErrorCode.RUN_NOT_FOUND,
                             f"Run {run_id!r} not found.", False,
                             "Call list_runs() to see available run IDs.")
        history = self._store.get_history(run_id, step_range=step_range)
        if fields:
            for obs in history:
                obs["metrics"] = {k: v for k, v in obs["metrics"].items() if k in fields}
        return self._ok(history)

    def get_sample(
        self,
        run_id: str,
        step: Optional[int] = None,
        n: int = 10,
    ) -> ToolResponse:
        """Return captured (prompt, completion, reward) tuples for a run.

        Args:
            run_id: Run identifier.
            step: If given, return samples near this step (within ±25 steps).
            n: Maximum number of samples to return.
        """
        run = self._store.get_run(run_id)
        if run is None:
            return self._err(ErrorCode.RUN_NOT_FOUND,
                             f"Run {run_id!r} not found.", False,
                             "Call list_runs() to see available run IDs.")
        samples = self._store.get_samples(run_id, limit=n if step is None else n * 10)
        if step is not None:
            samples = [s for s in samples if abs(s["step"] - step) <= 25][:n]
        return self._ok(samples)

    def compute_trend(
        self,
        run_id: str,
        metric: str,
        window: int,
    ) -> ToolResponse:
        """Compute slope, mean, and std for a metric over an arbitrary window.

        Args:
            run_id: Run identifier.
            metric: Metric key (e.g. "loss", "rewards/accuracy_reward/mean").
            window: Number of most-recent observation steps to include.
        """
        run = self._store.get_run(run_id)
        if run is None:
            return self._err(ErrorCode.RUN_NOT_FOUND,
                             f"Run {run_id!r} not found.", False,
                             "Call list_runs() to see available run IDs.")
        history = self._store.get_history(run_id)
        history = history[-window:] if len(history) > window else history
        vals = [h["metrics"].get(metric) for h in history]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        steps = [h["step"] for h in history if h["metrics"].get(metric) is not None]

        if len(vals) < 2:
            return self._ok({
                "metric": metric,
                "window": window,
                "computed_over_steps": [],
                "slope": None,
                "mean": None,
                "std": None,
                "note": "insufficient data",
            })

        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)

        # Least-squares slope.
        xs = list(range(n))
        xm = sum(xs) / n
        ym = mean
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, vals))
        den = sum((x - xm) ** 2 for x in xs)
        slope = num / den if den else 0.0

        return self._ok({
            "metric": metric,
            "window": window,
            "computed_over_steps": [steps[0], steps[-1]],
            "slope": slope,
            "mean": mean,
            "std": std,
        })

    def get_config(self, config_id: str) -> ToolResponse:
        """Return the full config dict for a config_id.

        Args:
            config_id: SHA-256[:16] hash of the config JSON.
        """
        config = self._store.get_config(config_id)
        if config is None:
            return self._err(
                ErrorCode.RUN_NOT_FOUND,
                f"Config {config_id!r} not found.",
                False,
                "config_id values come from run records or observation events.",
            )
        return self._ok(config)

    def get_eval(self, eval_id: str) -> ToolResponse:
        """Return the status and result of a previously-triggered evaluation.

        Args:
            eval_id: Identifier returned by eval().
        """
        with self._eval_lock:
            state = self._pending_evals.get(eval_id)

        if state is not None:
            if state["status"] == "running":
                return self._ok({"eval_id": eval_id, "status": "running"})
            if state["status"] == "done":
                return self._ok({
                    "eval_id": eval_id,
                    "status": "done",
                    "results": state["results"],
                })
            # failed
            return self._ok({
                "eval_id": eval_id,
                "status": "failed",
                "error": state.get("error"),
            })

        # Not in memory — try the store (completed in a prior session).
        # eval_id format: "eval-{run_id[:8]}-{timestamp}" — we query by run substring.
        # For simplicity in v0, return not-found with guidance.
        return self._err(
            ErrorCode.EVAL_NOT_FOUND,
            f"Eval {eval_id!r} not found in current session.",
            False,
            "Use get_run_details(run_id) to see all completed evals for a run.",
        )

    # =========================================================================
    # Agent integration helpers
    # =========================================================================

    def tool_specs(self) -> list[dict]:
        """Return Anthropic-format tool schemas for all 16 tools (14 + sleep + finalize).

        sleep and finalize are intercepted by AgentLoop; they appear here so the
        LLM sees them as callable tools.
        """
        from autolab.control_plane.specs import TOOL_SPECS
        return TOOL_SPECS

    def invoke_by_name(self, name: str, args: dict) -> "ToolResponse":
        """Dispatch a tool call by name. Used by AgentLoop to route LLM tool calls.

        sleep and finalize must be handled by AgentLoop before calling this method;
        they are not methods on ControlPlane.

        Raises:
            ValueError: if name is unknown or starts with '_'.
        """
        method = getattr(self, name, None)
        if method is None or name.startswith("_") or not callable(method):
            raise ValueError(f"Unknown tool: {name!r}")
        return method(**args)

    def get_pending_evals_snapshot(self) -> dict[str, str]:
        """Return a snapshot of eval_id → status for all pending evals.

        Used by SleepScheduler to detect eval completion without accessing
        private state directly.
        """
        with self._eval_lock:
            return {eid: info["status"] for eid, info in self._pending_evals.items()}
