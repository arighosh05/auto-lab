"""
Training callbacks for GRPO.

GRPOTrainer merges reward metrics into the logs dict before calling
TrainerCallback.on_log(). Keys logged by TRL 1.0.0:

  loss                              — policy gradient loss
  learning_rate                     — current LR
  reward                            — mean total reward (weighted sum)
  reward_std                        — std of total reward within groups
  rewards/<fn_name>/mean            — mean per reward function
  rewards/<fn_name>/std             — std per reward function
  grad_norm                         — gradient norm

Note: TRL 1.0.0 changed the key format from "rewards/<name>" (0.x)
to "rewards/<name>/mean". The per-function keys are:
  rewards/accuracy_reward/mean
  rewards/format_reward/mean
"""

import sys

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

_METRICS_LOG = "/workspace/metrics.log"


def _tprint(msg: str) -> None:
    """Write to both stderr (survives tqdm) and a dedicated metrics file."""
    print(msg, file=sys.stderr, flush=True)
    try:
        with open(_METRICS_LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


class LoggingCallback(TrainerCallback):
    """Human-readable training progress logger.

    Designed to be the primary console output during Phase 0.
    Replace with structured telemetry in later phases by swapping this
    callback out — the training loop itself has no inline prints.
    """

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

        step = state.global_step
        max_steps = state.max_steps or 0
        epoch = state.epoch or 0.0
        progress = f"{step / max_steps * 100:.1f}%" if max_steps > 0 else "?"

        loss = logs.get("loss", float("nan"))
        lr = logs.get("learning_rate", float("nan"))
        reward = logs.get("reward", float("nan"))
        reward_std = logs.get("reward_std", float("nan"))
        acc = logs.get("rewards/accuracy_reward/mean", float("nan"))
        fmt = logs.get("rewards/format_reward/mean", float("nan"))

        _tprint(
            f"[step {step:>6}/{max_steps} | {progress:>6} | epoch {epoch:.2f}]"
            f"  loss={loss:.4f}  lr={lr:.2e}"
            f"  reward={reward:.3f}±{reward_std:.3f}"
            f"  acc={acc:.3f}  fmt={fmt:.3f}"
        )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return
        _tprint(
            f"[step {state.global_step}] Checkpoint saved → "
            f"{args.output_dir}/checkpoint-{state.global_step}"
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return
        _tprint(
            f"\nTraining complete."
            f" Total steps: {state.global_step}."
            f" Output: {args.output_dir}"
        )
