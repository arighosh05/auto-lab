"""ManagedRun — wraps GRPOTrainer as a lifecycle-managed training run."""

import enum
import functools
import gc
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from autolab.trainer_pool.pause_callback import (
    HotModifyCallback,
    PauseCallback,
    StepTrackingCallback,
)

logger = logging.getLogger(__name__)


class RunStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"   # stopped cleanly with checkpoint saved; resumable
    DONE = "done"
    FAILED = "failed"
    KILLED = "killed"   # stopped permanently; not resumable


def _make_capturing_reward(fn, run_id: str, sample_queue: queue.Queue, capture_every: int = 1):
    """Wrap a reward function to capture (prompt, completion, reward) tuples.

    IMPORTANT: uses functools.wraps to preserve fn.__name__, which TRL uses
    to generate metric key names (e.g. rewards/accuracy_reward/mean).

    Args:
        fn: The reward function to wrap.
        run_id: Run identifier to tag captured samples.
        sample_queue: Queue to push captured samples into.
        capture_every: Capture one sample every N reward function calls.
    """
    call_count = [0]

    @functools.wraps(fn)
    def wrapper(prompts, completions, **kwargs):
        rewards = fn(prompts=prompts, completions=completions, **kwargs)
        call_count[0] += 1
        if call_count[0] % capture_every == 0 and prompts and completions:
            # Extract text from TRL completion format: list[list[dict]] or list[str]
            raw_completion = completions[0]
            if isinstance(raw_completion, list) and raw_completion:
                first = raw_completion[0]
                completion_text = first.get("content", "") if isinstance(first, dict) else str(first)
            else:
                completion_text = str(raw_completion)

            # Extract user message from conversational prompt format
            raw_prompt = prompts[0]
            if isinstance(raw_prompt, list):
                prompt_text = next(
                    (m.get("content", "") for m in reversed(raw_prompt) if m.get("role") == "user"),
                    "",
                )
            else:
                prompt_text = str(raw_prompt)

            reward_val = float(rewards[0]) if rewards else 0.0
            sample_queue.put_nowait({
                "run_id": run_id,
                "reward_fn": fn.__name__,
                "prompt": prompt_text,
                "completion": completion_text,
                "reward": reward_val,
            })
        return rewards

    return wrapper


@dataclass
class ManagedRun:
    """Manages the full lifecycle of a single GRPO training run.

    Args:
        run_id: Unique string identifier for this run.
        config: Parsed YAML config dict (full, not just grpo section).
        metrics_queue: Shared queue populated by MetricsHookCallback.
        sample_queue: Shared queue populated by reward function wrappers.
    """

    run_id: str
    config: dict
    metrics_queue: queue.Queue
    sample_queue: queue.Queue

    # ── Observable state ──────────────────────────────────────────────────────
    status: RunStatus = field(default=RunStatus.PENDING, init=False)
    start_time: Optional[float] = field(default=None, init=False)
    end_time: Optional[float] = field(default=None, init=False)
    final_metrics: dict = field(default_factory=dict, init=False)
    error: Optional[Exception] = field(default=None, init=False)

    # ── Control-plane signals (set by TrainerPool, polled by callbacks) ───────
    pause_requested: threading.Event = field(default_factory=threading.Event, init=False)
    kill_requested: threading.Event = field(default_factory=threading.Event, init=False)
    # Hot-modify dict: {param_name: new_value}; drained by HotModifyCallback.
    pending_mods: dict = field(default_factory=dict, init=False)

    # ── Checkpoint tracking ───────────────────────────────────────────────────
    # Set by PauseCallback.on_save when trainer saves a checkpoint on pause.
    checkpoint_path: Optional[str] = field(default=None, init=False)

    # ── Step tracking (for accurate audit log timestamps) ────────────────────
    # Updated by StepTrackingCallback at every on_step_end.
    current_step: int = field(default=0, init=False)
    # Updated by HotModifyCallback when it actually applies pending mods.
    last_modify_step: int = field(default=0, init=False)

    def start(self) -> None:
        """Build trainer, attach hooks, run trainer.train(). BLOCKS until done.

        On exit, status is set to one of:
          - DONE    — normal completion
          - FAILED  — unhandled exception during training
          - PAUSED  — stopped because pause_requested was set
          - KILLED  — stopped because kill_requested was set

        Failures are recorded in self.error. Does not re-raise exceptions so
        the caller (the pool's training thread) can inspect state cleanly.
        """
        self.status = RunStatus.RUNNING
        self.start_time = time.time()
        logger.info("[%s] Starting run", self.run_id)

        trainer = None
        try:
            # Lazy imports: trl/torch only needed when training actually starts.
            from autolab.trainer_pool.metrics_hook import MetricsHookCallback
            from autolab.training.grpo import build_trainer
            from autolab.training.rewards import accuracy_reward, format_reward

            # Wrap reward functions to capture samples into sample_queue.
            wrapped_reward_funcs = [
                _make_capturing_reward(accuracy_reward, self.run_id, self.sample_queue),
                _make_capturing_reward(format_reward, self.run_id, self.sample_queue),
            ]

            trainer = build_trainer(self.config, reward_funcs=wrapped_reward_funcs)

            # Metrics + control-plane callbacks. Order matters: PauseCallback
            # and HotModifyCallback run after MetricsHookCallback.
            trainer.add_callback(MetricsHookCallback(
                run_id=self.run_id,
                metrics_queue=self.metrics_queue,
            ))
            trainer.add_callback(StepTrackingCallback(run=self))
            trainer.add_callback(PauseCallback(run=self))
            trainer.add_callback(HotModifyCallback(run=self))

            trainer.train()

            if trainer.state.log_history:
                self.final_metrics = trainer.state.log_history[-1]

            # Determine why training exited and set status accordingly.
            if self.kill_requested.is_set():
                self.status = RunStatus.KILLED
                logger.info("[%s] Run killed (checkpoint saved)", self.run_id)
            elif self.pause_requested.is_set():
                self.status = RunStatus.PAUSED
                logger.info(
                    "[%s] Run paused at checkpoint: %s",
                    self.run_id, self.checkpoint_path,
                )
            else:
                # Normal completion: save final model.
                logger.info("[%s] Saving final model", self.run_id)
                trainer.save_model()
                trainer.save_state()
                self.status = RunStatus.DONE
                logger.info("[%s] Run completed successfully", self.run_id)

        except Exception as exc:
            self.error = exc
            self.status = RunStatus.FAILED
            logger.exception("[%s] Run failed: %s", self.run_id, exc)

        finally:
            self.end_time = time.time()
            # Explicit cleanup: free model/optimizer GPU memory so the next run
            # in the same process starts with a clean allocator.  Python's GC
            # will not immediately collect 'trainer' when an OOM exception is
            # raised because the traceback holds a reference to the frame.
            if trainer is not None:
                del trainer
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
