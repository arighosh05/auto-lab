"""
Evaluation harness for GRPO-trained checkpoints.

run_eval() loads a checkpoint, runs greedy inference on a subset of the
MATH test split, and reports accuracy broken down by difficulty level and
subject type. Results are returned as a dict and optionally saved as JSON.

Intentionally separate from the training loop so it can be run as a
standalone script, in CI, or as an agent-triggered evaluation in later phases.
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_NEEDLE = "\\boxed{"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step, showing "
    "your reasoning clearly. At the end of your solution, write your final "
    "answer in a LaTeX box using the format: \\boxed{your answer here}. "
    "Do not include units or extra text inside the box — only the mathematical "
    "expression or value."
)


def _extract_last_boxed(text: str) -> str:
    """Extract the last \\boxed{...} from text using brace counting."""
    idx = 0
    last_result = ""
    while True:
        start = text.find(_NEEDLE, idx)
        if start == -1:
            break
        depth = 0
        end = start
        for j in range(start + len(_NEEDLE) - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        content = text[start + len(_NEEDLE):end]
        last_result = f"\\boxed{{{content}}}"
        idx = start + 1
    return last_result if last_result else text


def _strip_think_tags(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _is_correct(prediction: str, gold_answer: str) -> bool:
    """Check correctness using math-verify symbolic comparison."""
    cfg = [LatexExtractionConfig()]
    try:
        pred_parsed = parse(prediction, extraction_config=cfg)
        gold_parsed = parse(gold_answer, extraction_config=cfg)
        if pred_parsed and gold_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False


def run_eval(
    checkpoint_path: str,
    config: Optional[dict] = None,
    n_samples: int = 200,
    output_path: Optional[str] = None,
    dataset_name: str = "hendrycks/competition_math",
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
) -> dict:
    """Evaluate a trained checkpoint on the MATH test set.

    Args:
        checkpoint_path: Path to a saved model checkpoint directory.
        config: Optional full YAML config dict. Values from its 'eval' section
                override the defaults for batch_size, max_new_tokens, n_samples.
                Values for dataset_name and system_prompt are also read if present.
        n_samples: Number of test samples to evaluate. -1 = full test set.
        output_path: If given, saves results JSON here.
        dataset_name: HuggingFace dataset identifier.
        system_prompt: System message used during inference.
        batch_size: Inference batch size.
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        Dict with keys: overall_accuracy, by_level, by_type,
                        n_correct, n_samples, checkpoint.
    """
    # Override defaults from config
    if config:
        dataset_name = config.get("dataset_name", dataset_name)
        system_prompt = config.get("system_prompt", system_prompt).strip()
        if "eval" in config:
            ec = config["eval"]
            n_samples = ec.get("n_samples", n_samples)
            batch_size = ec.get("batch_size", batch_size)
            max_new_tokens = ec.get("max_new_tokens", max_new_tokens)

    # Resolve to absolute path — transformers 5.x rejects relative paths as invalid repo IDs.
    checkpoint_path = os.path.abspath(checkpoint_path)
    logger.info("Loading checkpoint: %s", checkpoint_path)

    # Checkpoints saved with save_only_model=True contain model weights + config.json
    # but no tokenizer files.  Load tokenizer from the base model name in that case.
    tokenizer_source = checkpoint_path
    if not os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
        tokenizer_source = (config or {}).get("model_name", checkpoint_path)
        logger.info("No tokenizer in checkpoint; loading tokenizer from: %s", tokenizer_source)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    logger.info("Loading test split: %s", dataset_name)
    test_data = load_dataset(dataset_name, split="test")
    if n_samples > 0 and n_samples < len(test_data):
        test_data = test_data.select(range(n_samples))
    logger.info("Evaluating on %d samples...", len(test_data))

    # Pre-extract ground truth answers (last \boxed{} from solution)
    ground_truths = [_extract_last_boxed(ex["solution"]) for ex in test_data]

    # Batched greedy inference
    all_predictions: list[str] = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data.select(range(i, min(i + batch_size, len(test_data))))

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["problem"]},
            ]
            for ex in batch
        ]

        texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: disable think tags for clean greedy eval
            )
            for msgs in messages_batch
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only generated tokens (strip prompt)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )
        all_predictions.extend(decoded)

        if (i // batch_size) % 10 == 0:
            logger.info("  %d/%d evaluated", i + len(batch), len(test_data))

    # Score
    n_correct = 0
    level_correct: dict[str, int] = defaultdict(int)
    level_total: dict[str, int] = defaultdict(int)
    type_correct: dict[str, int] = defaultdict(int)
    type_total: dict[str, int] = defaultdict(int)

    for pred_text, gt, ex in zip(all_predictions, ground_truths, test_data):
        level = ex["level"]
        subj = ex["type"]

        # Strip think tags then extract answer
        clean = _strip_think_tags(pred_text)
        pred_answer = _extract_last_boxed(clean)
        correct = _is_correct(pred_answer, gt)

        if correct:
            n_correct += 1
            level_correct[level] += 1
            type_correct[subj] += 1
        level_total[level] += 1
        type_total[subj] += 1

    total = len(test_data)
    overall_accuracy = n_correct / total if total else 0.0

    by_level = {lvl: level_correct[lvl] / level_total[lvl] for lvl in sorted(level_total)}
    by_type = {t: type_correct[t] / type_total[t] for t in sorted(type_total)}

    results = {
        "overall_accuracy": overall_accuracy,
        "by_level": by_level,
        "by_type": by_type,
        "n_correct": n_correct,
        "n_samples": total,
        "checkpoint": str(checkpoint_path),
    }

    logger.info("Overall accuracy: %.4f (%d/%d)", overall_accuracy, n_correct, total)
    for lvl, acc in by_level.items():
        logger.info("  %s: %.4f", lvl, acc)
    for t, acc in by_type.items():
        logger.info("  %s: %.4f", t, acc)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results
