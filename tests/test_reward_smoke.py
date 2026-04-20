"""Smoke test for accuracy_reward — verifies the reward function scores
known-correct answers as 1.0.  No GPU needed; runs on CPU in ~30 seconds.

Usage:
    python tests/test_reward_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset

from autolab.training.data import _extract_last_boxed
from autolab.training.rewards import accuracy_reward


def _wrap(text: str) -> list[dict]:
    """Wrap a completion string in TRL's conversational format."""
    return [{"role": "assistant", "content": text}]


def main() -> int:
    print("Loading MATH-lighteval Level 1 test split (first 10 problems)...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    level1 = [ex for ex in ds if ex["level"] == "Level 1"][:10]
    print(f"Loaded {len(level1)} Level 1 problems.\n")

    passed = 0
    failed = 0

    header = f"{'#':<3} {'Gold answer':<30} {'Format':<12} {'Reward':<8} {'Status'}"
    print(header)
    print("-" * len(header))

    for i, ex in enumerate(level1):
        # gold is now the bare inner value (no \boxed{} wrapper)
        gold = _extract_last_boxed(ex["solution"])
        if not gold:
            print(f"{i:<3} (no gold answer found — skipping)")
            continue

        formats = {
            "exact":    _wrap(f"\\boxed{{{gold}}}"),
            "reasoned": _wrap(f"Step 1: set up equation. Step 2: simplify. Therefore \\boxed{{{gold}}}"),
            "thinking": _wrap(f"<think>Let me work through this problem carefully.\n"
                              f"The answer is {gold}.\n</think>\n\\boxed{{{gold}}}"),
        }

        for fmt_name, completion in formats.items():
            rewards = accuracy_reward(
                completions=[completion],
                answer=[gold],
            )
            r = rewards[0]
            status = "PASS" if r == 1.0 else "FAIL"
            if r == 1.0:
                passed += 1
            else:
                failed += 1
                print(f"  Problem: {ex['problem'][:80]!r}")
                print(f"  Gold: {gold!r}")
                print(f"  Completion: {completion[0]['content'][:120]!r}")
            print(f"{i:<3} {gold[:28]:<30} {fmt_name:<12} {r:<8.1f} {status}")

    total = passed + failed
    print(f"\n{'='*60}")
    rate = passed / total * 100 if total > 0 else 0
    print(f"Pass rate: {passed}/{total} ({rate:.0f}%)")
    verdict = "PROCEED" if failed == 0 else "INVESTIGATE"
    print(f"Verdict: {verdict}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
