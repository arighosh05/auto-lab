#!/usr/bin/env python3
"""
Evaluation entry point for autolab Phase 0 GRPO.

Usage:
    python scripts/eval.py --checkpoint outputs/grpo-qwen3-1.7b-math/checkpoint-200
    python scripts/eval.py --checkpoint <path> --n-samples 500 --output results/step200.json
    python scripts/eval.py --checkpoint <path> --n-samples -1  # full test set
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolab.eval.evaluator import run_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a GRPO checkpoint on MATH.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory.")
    p.add_argument("--n-samples", type=int, default=200, help="Test samples to use (-1 = all).")
    p.add_argument("--output", default=None, help="Save results JSON to this path.")
    p.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path to read eval defaults from.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    results = run_eval(
        checkpoint_path=args.checkpoint,
        config=config,
        n_samples=args.n_samples,
        output_path=args.output,
    )

    print("\n=== Evaluation Results ===")
    print(f"Checkpoint : {results['checkpoint']}")
    print(f"Correct    : {results['n_correct']}/{results['n_samples']}")
    print(f"Accuracy   : {results['overall_accuracy']:.4f}")

    print("\nBy level:")
    for level, acc in results["by_level"].items():
        bar = "#" * int(acc * 20)
        print(f"  {level:<10} {acc:.4f}  {bar}")

    print("\nBy type:")
    for subj, acc in results["by_type"].items():
        print(f"  {subj:<25} {acc:.4f}")

    if args.output:
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
