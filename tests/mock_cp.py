"""MockControlPlane — stubbed ControlPlane for Phase A (mock dry run) testing.

All 14 tools return canned ToolResponse objects. No real training, no SQLite,
no GPU. Used to test AgentLoop wiring without infrastructure dependencies.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from autolab.control_plane.specs import TOOL_SPECS
from autolab.control_plane.types import StatusBar, ToolError, ToolResponse


def _ok(result: Any, **status_kwargs) -> ToolResponse:
    return ToolResponse(status=_status(**status_kwargs), result=result)


def _err(code: str, message: str, suggested: str, **status_kwargs) -> ToolResponse:
    return ToolResponse(
        status=_status(**status_kwargs),
        error=ToolError(code=code, message=message, retryable=False, suggested_action=suggested),
    )


def _status(
    active_run_id: Optional[str] = None,
    paused_run_ids: Optional[list] = None,
    pending_evals: Optional[list] = None,
    gpu_state: str = "idle",
) -> StatusBar:
    return StatusBar(
        active_run_id=active_run_id,
        paused_run_ids=paused_run_ids or [],
        pending_evals=pending_evals or [],
        gpu_state=gpu_state,
    )


class MockControlPlane:
    """Stubbed ControlPlane that returns scripted responses."""

    def __init__(self) -> None:
        self._active_run_id: Optional[str] = None
        self._runs: dict[str, dict] = {}
        self._evals: dict[str, str] = {}  # eval_id → status
        self._eval_lock = threading.Lock()

    # ── Required public methods ───────────────────────────────────────────────

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    def invoke_by_name(self, name: str, args: dict) -> ToolResponse:
        method = getattr(self, name, None)
        if method is None:
            return _err("UNKNOWN_TOOL", f"Unknown: {name}", "Check tool name")
        return method(**args)

    def get_pending_evals_snapshot(self) -> dict[str, str]:
        with self._eval_lock:
            return dict(self._evals)

    def _status_bar(self) -> StatusBar:
        return _status(
            active_run_id=self._active_run_id,
            paused_run_ids=[r for r, d in self._runs.items() if d["status"] == "paused"],
            pending_evals=[e for e, s in self._evals.items() if s == "running"],
            gpu_state="training" if self._active_run_id else "idle",
        )

    # ── Write tools ───────────────────────────────────────────────────────────

    def start_run(self, config: dict, reason: str = "") -> ToolResponse:
        run_id = f"mock-run-{int(time.time())}"
        self._runs[run_id] = {"status": "running", "step": 0}
        self._active_run_id = run_id
        return ToolResponse(
            status=self._status_bar(),
            result={"run_id": run_id, "config_id": "mock-config-abc123"},
        )

    def fork(self, parent_run_id: str, overrides: dict, reason: str = "") -> ToolResponse:
        if parent_run_id not in self._runs:
            return _err("RUN_NOT_FOUND", f"Not found: {parent_run_id}", "Use list_runs()")
        child_id = f"mock-fork-{int(time.time())}"
        self._runs[child_id] = {"status": "running", "step": 5}
        self._runs[parent_run_id]["status"] = "paused"
        self._active_run_id = child_id
        return ToolResponse(
            status=self._status_bar(),
            result={"run_id": child_id, "parent_run_id": parent_run_id, "fork_step": 5, "config_id": "mock-fork-cfg"},
        )

    def kill(self, run_id: str, reason: str = "") -> ToolResponse:
        if run_id in self._runs:
            self._runs[run_id]["status"] = "killed"
            if self._active_run_id == run_id:
                self._active_run_id = None
        return ToolResponse(status=self._status_bar(), result={"run_id": run_id, "status": "killed"})

    def modify(self, run_id: str, overrides: dict, reason: str = "") -> ToolResponse:
        if run_id not in self._runs:
            return _err("RUN_NOT_FOUND", f"Not found: {run_id}", "Use list_runs()")
        return ToolResponse(
            status=self._status_bar(),
            result={"run_id": run_id, "new_config_id": "mock-modified-cfg", "applied": overrides, "step": 10},
        )

    def set_active(self, run_id: Optional[str], reason: str = "") -> ToolResponse:
        self._active_run_id = run_id
        return ToolResponse(status=self._status_bar(), result={"active_run_id": run_id})

    def set_cadence(self, run_id: str, observation_cadence: int, sample_cadence: int, reason: str = "") -> ToolResponse:
        return ToolResponse(
            status=self._status_bar(),
            result={"observation_cadence": observation_cadence, "sample_cadence": sample_cadence},
        )

    def eval(self, run_id: str, benchmark: str, n_samples: int, reason: str = "") -> ToolResponse:
        eval_id = f"eval-{run_id[:8]}-{int(time.time())}"
        with self._eval_lock:
            self._evals[eval_id] = "running"
        # Simulate completion after 2s in background
        def _complete():
            time.sleep(2)
            with self._eval_lock:
                self._evals[eval_id] = "done"
        threading.Thread(target=_complete, daemon=True).start()
        return ToolResponse(
            status=self._status_bar(),
            result={"eval_id": eval_id, "status": "running", "estimated_seconds": 30, "training_paused": True},
        )

    # ── Read tools ────────────────────────────────────────────────────────────

    def get_run_details(self, run_id: str) -> ToolResponse:
        if run_id not in self._runs:
            return _err("RUN_NOT_FOUND", f"Not found: {run_id}", "Use list_runs()")
        return ToolResponse(status=self._status_bar(), result={"run_id": run_id, **self._runs[run_id]})

    def list_runs(self, status_filter: Optional[list] = None) -> ToolResponse:
        runs = [
            {"run_id": rid, **d}
            for rid, d in self._runs.items()
            if status_filter is None or d["status"] in status_filter
        ]
        return ToolResponse(status=self._status_bar(), result=runs)

    def get_history(self, run_id: str, step_range: Optional[list] = None, fields: Optional[list] = None) -> ToolResponse:
        return ToolResponse(status=self._status_bar(), result=[])

    def get_sample(self, run_id: str, step: Optional[int] = None, n: int = 5) -> ToolResponse:
        return ToolResponse(status=self._status_bar(), result=[])

    def compute_trend(self, run_id: str, metric: str, window: int = 20) -> ToolResponse:
        return ToolResponse(
            status=self._status_bar(),
            result={"slope": 0.0, "mean": 0.0, "std": 0.0, "computed_over_steps": 0},
        )

    def get_config(self, config_id: str) -> ToolResponse:
        return ToolResponse(status=self._status_bar(), result={"config_id": config_id, "grpo": {}})

    def get_eval(self, eval_id: str) -> ToolResponse:
        with self._eval_lock:
            status = self._evals.get(eval_id, "not_found")
        if status == "not_found":
            return _err("EVAL_NOT_FOUND", f"Not found: {eval_id}", "Check eval_id")
        result: dict = {"eval_id": eval_id, "status": status}
        if status == "done":
            result["accuracy"] = 0.42
            result["by_level"] = {"1": 0.55, "2": 0.40, "3": 0.28}
        return ToolResponse(status=self._status_bar(), result=result)
