"""MetricsHookCallback — captures TRL log events into a queue.Queue."""

import logging
import queue

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Keys we attempt to extract from TRL's logs dict.
# All are optional — missing or non-numeric values are silently skipped.
# Includes both old (pre-1.0) and new (1.0+) TRL key formats.
_WANTED_KEYS = frozenset([
    "loss",
    "learning_rate",
    "reward",
    "reward_std",
    "rewards/accuracy_reward/mean",
    "rewards/format_reward/mean",
    "rewards/accuracy_reward",
    "rewards/format_reward",
    "kl",
    "entropy",
    "grad_norm",
    "completion_length",
    "mean_completion_length",
])


class MetricsHookCallback(TrainerCallback):
    """Intercepts TRL on_log events and enqueues metric dicts.

    Runs alongside LoggingCallback — does not replace it. Never mutates
    TrainerControl. Thread-safe: queue.Queue is designed for cross-thread use.

    Args:
        run_id: Identifier for this training run (tags every queue entry).
        metrics_queue: Shared queue.Queue. Entries are dicts:
            {"run_id": str, "step": int, "metrics": dict[str, float]}
    """

    def __init__(self, run_id: str, metrics_queue: queue.Queue) -> None:
        super().__init__()
        self.run_id = run_id
        self._queue = metrics_queue

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero or logs is None:
            return

        extracted: dict[str, float] = {}
        for key in _WANTED_KEYS:
            if key in logs:
                val = logs[key]
                try:
                    extracted[key] = float(val)
                except (TypeError, ValueError):
                    logger.debug(
                        "MetricsHookCallback: skipping non-numeric key %r = %r", key, val
                    )

        if extracted:
            self._queue.put_nowait({
                "run_id": self.run_id,
                "step": state.global_step,
                "metrics": extracted,
            })
