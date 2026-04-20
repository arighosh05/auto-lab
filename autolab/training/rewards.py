"""
Math reward functions for GRPO training on competition mathematics.

TRL calls reward functions with:
    reward_func(prompts=prompts, completions=completions, **dataset_columns)

When the dataset uses conversational format (prompt is list[dict]),
TRL wraps completions as list[list[dict]], where each element is:
    [{"role": "assistant", "content": "<the generated text>"}]

Both reward functions handle this by extracting the content string first.
Qwen3 in thinking mode emits <think>...</think> blocks before its answer.
We strip those before passing to math-verify and before checking for \boxed{}.

The 'answer' kwarg is populated automatically from the dataset's 'answer'
column by TRL (non-prompt/completion columns flow through as kwargs).

Gold answers ('answer' column) are bare inner values, e.g. "-4" or "\\frac{1}{2}".
Both gold and pred are wrapped in \\boxed{} before calling parse() so that
LatexExtractionConfig reliably extracts the expression on both sides.
"""

import json
import os
import re
from typing import Union

from math_verify import parse, verify

from autolab.training.data import _extract_last_boxed

_TRACE_PATH = "/tmp/reward_trace.jsonl"
_TRACE_LIMIT = 32  # capture first 32 calls (= 2 full 16-completion batches) then stop

# Matches <think>...</think> including multiline content
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Matches \boxed{ — used by format_reward to check presence
_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_text(completion: Union[str, list]) -> str:
    """Extract plain text from a completion regardless of TRL's wrapping format.

    TRL passes completions as:
    - str: when prompt column is plain text
    - list[dict]: when prompt column is conversational; element is
      [{"role": "assistant", "content": "..."}]
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get("content", "")
    return ""


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 thinking-mode output.

    Handles two cases:
    - Complete block: <think>...</think> is present — stripped entirely.
    - Truncated block: <think> with no closing </think> (completion cut off at
      max_completion_length) — everything from <think> onward is removed,
      leaving an empty string. parse() on "" returns [] → reward 0.0, which is
      correct: the model never surfaced a final answer.
    """
    # Strip complete blocks
    stripped = _THINK_RE.sub("", text)
    # Handle unclosed <think> (truncated completion)
    open_idx = stripped.find("<think>")
    if open_idx != -1:
        stripped = stripped[:open_idx]
    return stripped.strip()


def accuracy_reward(
    completions: list[Union[str, list]],
    answer: list[str],
    **kwargs,
) -> list[float]:
    """Compute math accuracy reward using symbolic verification.

    Uses math-verify to compare the model's boxed answer against the
    ground truth. Returns 1.0 for a correct answer, 0.0 otherwise.

    Args:
        completions: Model completions from TRL. Each is either a str or
            [{"role": "assistant", "content": ...}].
        answer: Bare ground-truth values (inner content of \\boxed{}, no wrapper),
            one per completion.
        **kwargs: Other dataset columns (level, type) — passed through, ignored.

    Returns:
        List of float rewards, length == len(completions).
    """
    rewards = []

    # Count existing trace lines to enforce the cap without re-reading the file.
    try:
        with open(_TRACE_PATH) as _f:
            _trace_count = sum(1 for _ in _f)
    except FileNotFoundError:
        _trace_count = 0
    _do_trace = _trace_count < _TRACE_LIMIT

    for completion, gt_answer in zip(completions, answer):
        text = _strip_think_tags(_extract_text(completion))
        bucket = "unknown"
        try:
            # Extract only the last \boxed{} inner content from the stripped text.
            # This avoids calling parse() on the full completion (slow / hangs on
            # long CoT with many intermediate \boxed{} expressions).
            pred_inner = _extract_last_boxed(text)
            if not pred_inner:
                bucket = "no_pred_box"
                rewards.append(0.0)
            else:
                # Wrap both sides in \boxed{} and use default parse() config.
                # LatexExtractionConfig requires surrounding context and returns []
                # for bare numbers; the default handles integers, fractions, etc.
                pred_parsed = parse(f"\\boxed{{{pred_inner}}}", parsing_timeout=None)
                gold_parsed = parse(f"\\boxed{{{gt_answer}}}", parsing_timeout=None)

                if not gold_parsed:
                    bucket = "gold_parse_fail"
                elif not pred_parsed:
                    bucket = "pred_parse_fail"
                elif verify(gold_parsed, pred_parsed, timeout_seconds=None):
                    bucket = "correct"
                else:
                    bucket = "wrong_answer"

                # verify() is asymmetric: gold first, prediction second
                rewards.append(1.0 if bucket == "correct" else 0.0)
        except Exception as _e:
            bucket = f"exception:{type(_e).__name__}"
            rewards.append(0.0)

        if _do_trace and _trace_count < _TRACE_LIMIT:
            try:
                with open(_TRACE_PATH, "a") as _tf:
                    _tf.write(json.dumps({
                        "bucket": bucket,
                        "gold": gt_answer,
                        "pred": _extract_last_boxed(text) if text else "",
                    }) + "\n")
                _trace_count += 1
            except Exception:
                pass

    return rewards


def format_reward(
    completions: list[Union[str, list]],
    **kwargs,
) -> list[float]:
    """Reward for producing \boxed{} in the final answer (outside think tags).

    Returns 1.0 if the completion contains \boxed{} outside of <think> blocks,
    0.0 otherwise. A \boxed{} only inside <think> means the model hasn't learned
    to surface the answer in the required output format.

    Args:
        completions: Model completions from TRL.
        **kwargs: Ignored.

    Returns:
        List of float rewards, length == len(completions).
    """
    rewards = []
    for completion in completions:
        text = _strip_think_tags(_extract_text(completion))
        rewards.append(1.0 if _BOXED_RE.search(text) else 0.0)
    return rewards
