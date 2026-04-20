"""PauseCallback and HotModifyCallback — training-loop hooks for control plane."""

from __future__ import annotations

import logging
import re as _re
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from transformers import TrainerCallback, TrainerControl
except ImportError:  # allow import without GPU environment for smoke tests
    class TrainerCallback:  # type: ignore[no-redef]
        pass
    class TrainerControl:  # type: ignore[no-redef]
        should_save: bool = False
        should_training_stop: bool = False

if TYPE_CHECKING:
    from autolab.trainer_pool.runner import ManagedRun

logger = logging.getLogger(__name__)

# Parameters that can be hot-modified without restarting the trainer.
_HOT_MODIFY_ALLOWLIST = frozenset(
    {"learning_rate", "beta", "epsilon", "temperature", "top_p", "top_k"}
)


class PauseCallback(TrainerCallback):
    """Signals TRL to save a checkpoint then stop training.

    The TrainerPool calls ``run.pause_requested.set()`` or
    ``run.kill_requested.set()``.  This callback checks both events at every
    ``on_step_end``.  When either is set it requests a checkpoint save and a
    training stop, so the training loop exits cleanly after the current step.

    ``on_save`` records the checkpoint path on the run by globbing the
    output_dir for the highest-numbered checkpoint directory.  Reading from disk
    is correct regardless of TRL's internal naming conventions or gradient-
    accumulation edge cases; computing the path from ``state.global_step`` is
    brittle.
    """

    def __init__(self, run: "ManagedRun") -> None:
        self._run = run

    def on_step_end(
        self, args, state, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if self._run.pause_requested.is_set() or self._run.kill_requested.is_set():
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control: TrainerControl, **kwargs) -> TrainerControl:
        # Read what's actually on disk — correct regardless of TRL naming.
        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            [p for p in output_dir.glob("checkpoint-*")
             if p.is_dir() and _re.fullmatch(r"checkpoint-\d+", p.name)],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            self._run.checkpoint_path = str(checkpoints[-1])
            logger.debug(
                "PauseCallback: recorded checkpoint_path=%s for run %s",
                self._run.checkpoint_path,
                self._run.run_id,
            )
        return control


class HotModifyCallback(TrainerCallback):
    """Applies pending config modifications at the start of each step.

    The control plane pushes ``{param: new_value}`` into ``run.pending_mods``.
    This callback drains the dict atomically at ``on_step_begin`` and applies
    each modification directly to the objects TRL passes via ``**kwargs``.

    No reference to the trainer object is stored — everything needed is
    available in the standard callback signature:
    - ``args``      TrainingArguments / GRPOConfig  → beta, epsilon
    - ``kwargs["optimizer"]``                       → learning_rate
    - ``kwargs["model"]``                           → generation_config fields

    Allowlist: learning_rate, beta, epsilon, temperature, top_p, top_k.
    Unknown params are logged as warnings and skipped.
    """

    def __init__(self, run: "ManagedRun") -> None:
        self._run = run

    def on_step_begin(
        self, args, state, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if not self._run.pending_mods:
            return control

        # Drain atomically — swap out the dict so concurrent writes in the main
        # thread don't race with our iteration.
        mods, self._run.pending_mods = self._run.pending_mods, {}

        optimizer = kwargs.get("optimizer")
        model = kwargs.get("model")

        for param, value in mods.items():
            if param not in _HOT_MODIFY_ALLOWLIST:
                logger.warning(
                    "HotModifyCallback: param %r not in allowlist — skipped (run=%s)",
                    param, self._run.run_id,
                )
                continue
            try:
                if param == "learning_rate":
                    if optimizer is not None:
                        for pg in optimizer.param_groups:
                            pg["lr"] = value
                        # Reset scheduler base_lrs so lr_scheduler.step() on the
                        # next step doesn't override the new value.  The cosine
                        # schedule computes lr = base_lr * f(step); updating base_lrs
                        # here means all future steps decay from the new base.
                        lr_scheduler = kwargs.get("lr_scheduler")
                        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
                            lr_scheduler.base_lrs = [value] * len(lr_scheduler.base_lrs)
                            logger.info(
                                "HotModifyCallback: reset lr_scheduler.base_lrs to %s",
                                value,
                            )
                        else:
                            logger.warning(
                                "HotModifyCallback: lr_scheduler not available or has no"
                                " base_lrs — LR will be overridden by scheduler next step"
                            )
                    else:
                        logger.warning(
                            "HotModifyCallback: optimizer not available at step_begin"
                            " — learning_rate not applied"
                        )
                elif param in ("beta", "epsilon"):
                    setattr(args, param, value)
                elif param in ("temperature", "top_p", "top_k"):
                    if model is not None and hasattr(model, "generation_config"):
                        setattr(model.generation_config, param, value)
                    else:
                        logger.warning(
                            "HotModifyCallback: model.generation_config not available"
                            " — %s not applied", param
                        )
            except Exception:
                logger.exception(
                    "HotModifyCallback: failed to apply param %r=%r", param, value
                )

        logger.info(
            "HotModifyCallback: applied %s at step %d (run=%s)",
            list(mods.keys()), state.global_step, self._run.run_id,
        )
        # Record actual application step for modify audit log precision.
        self._run.last_modify_step = state.global_step
        return control


class StepTrackingCallback(TrainerCallback):
    """Updates run.current_step at every on_step_end.

    Keeps ManagedRun.current_step within one step of the live trainer state so
    the control plane can stamp modifications with an accurate step number.
    """

    def __init__(self, run: "ManagedRun") -> None:
        self._run = run

    def on_step_end(self, args, state, control: TrainerControl, **kwargs) -> TrainerControl:
        self._run.current_step = state.global_step
        return control
