"""Response types and error codes for the control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ── Error codes ───────────────────────────────────────────────────────────────

class ErrorCode:
    """Stable string codes for control-plane errors.

    Design notes on intentional absences:
    - ``RUN_ALREADY_ACTIVE`` is absent.  ``set_active(run_id)`` is idempotent:
      if the run is already active it returns success, not an error.
    - ``CADENCE_OUT_OF_BOUNDS`` is absent.  ``set_cadence()`` clamps values
      silently and returns the actually-applied values.  This is intentional
      per the v1 schema-lock design.  Do not add this code without revisiting
      the spec.
    """

    RUN_NOT_FOUND           = "RUN_NOT_FOUND"
    RUN_NOT_FORKABLE        = "RUN_NOT_FORKABLE"       # status is failed/killed
    RUN_NOT_RESUMABLE       = "RUN_NOT_RESUMABLE"      # not in paused state
    PARAM_NOT_MODIFIABLE    = "PARAM_NOT_MODIFIABLE"   # not in hot-modify allowlist
    PARAM_OUT_OF_RANGE      = "PARAM_OUT_OF_RANGE"
    NO_CHECKPOINT           = "NO_CHECKPOINT"
    CHECKPOINT_COPY_FAILED  = "CHECKPOINT_COPY_FAILED"
    TRAINER_START_FAILED    = "TRAINER_START_FAILED"
    EVAL_LAUNCH_FAILED      = "EVAL_LAUNCH_FAILED"
    EVAL_NOT_FOUND          = "EVAL_NOT_FOUND"
    BENCHMARK_UNKNOWN       = "BENCHMARK_UNKNOWN"
    PAUSE_TIMEOUT           = "PAUSE_TIMEOUT"
    INVALID_CONFIG          = "INVALID_CONFIG"
    BUDGET_EXHAUSTED        = "BUDGET_EXHAUSTED"       # session compute budget exceeded
    SESSION_FINALIZED       = "SESSION_FINALIZED"      # session has been finalized; no more ops


# ── Response shapes ───────────────────────────────────────────────────────────

@dataclass
class StatusBar:
    """Ambient system state included in every tool response.

    Attributes:
        active_run_id: run_id currently consuming the GPU, or None.
        paused_run_ids: run_ids that are stopped with checkpoints saved.
        pending_evals: eval_ids of evaluations currently in progress.
        gpu_state: "training" | "evaluating" | "idle"
        budget_remaining_seconds: seconds remaining in the session budget, or None
            if no budget is active (e.g. called outside an agent session).
        warnings: ambient warnings the agent should read (e.g. "budget 90% consumed").
    """
    active_run_id: Optional[str]
    paused_run_ids: list[str]
    pending_evals: list[str]
    gpu_state: str  # "training" | "evaluating" | "idle"
    budget_remaining_seconds: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ToolError:
    """Structured error returned when a control-plane operation fails.

    Attributes:
        code: Stable string from :class:`ErrorCode`.
        message: Human-readable description of what went wrong.
        retryable: True if retrying with the same arguments could succeed
            (e.g. transient IO error).  False for logical failures.
        suggested_action: What the agent should do instead.
    """
    code: str
    message: str
    retryable: bool
    suggested_action: str


@dataclass
class ToolResponse:
    """Envelope returned by every control-plane tool.

    Exactly one of ``result`` or ``error`` is non-None on any given response.

    Attributes:
        status: Current system state (always present, even on error).
        result: Success payload — shape varies per tool.
        error: Structured error (present only on failure).
    """
    status: StatusBar
    result: Optional[Any] = None
    error: Optional[ToolError] = None
