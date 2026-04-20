"""
Dataset preprocessing for GRPO training on hendrycks/competition_math.

Produces a HuggingFace Dataset with columns:
  - prompt: list[dict]  — conversational format [system, user] messages
  - answer: str         — ground truth answer as a \\boxed{} string
  - level: str          — "Level 1" through "Level 5"
  - type: str           — "Algebra", "Number Theory", etc.

TRL passes 'answer', 'level', 'type' as kwargs to all reward functions
automatically because remove_unused_columns=False.

Ground truth extraction:
    MATH solutions box intermediate results as they go, then box the final
    answer last. We always extract the LAST \\boxed{} in the solution string.
    Grabbing the first would silently corrupt the reward signal.
"""

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

_NEEDLE = "\\boxed{"


def _extract_last_boxed(solution: str) -> str:
    """Extract the last \\boxed{...} from a solution using brace counting.

    Regex cannot reliably handle arbitrary nesting (e.g. \\boxed{\\frac{\\sqrt{6}}{4}}).
    We scan forward from each \\boxed{ token, count braces to find the matching },
    and return the last one found. This correctly handles any nesting depth.

    MATH solutions frequently box intermediate results mid-proof; we always want
    the LAST \\boxed{} which is the final answer.

    Returns the inner content string (without \\boxed{} wrapper), or "" if not found.
    """
    idx = 0
    last_result = ""
    while True:
        start = solution.find(_NEEDLE, idx)
        if start == -1:
            break
        depth = 0
        end = start
        for j in range(start + len(_NEEDLE) - 1, len(solution)):
            if solution[j] == "{":
                depth += 1
            elif solution[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        content = solution[start + len(_NEEDLE):end]
        last_result = content
        idx = start + 1

    if not last_result:
        logger.warning(
            "No \\boxed{} found in solution (first 120 chars): %.120s", solution
        )
    return last_result


def build_dataset(
    dataset_name: str,
    split: str,
    system_prompt: str,
    seed: int = 42,
    max_level: int = 5,
) -> Dataset:
    """Load and preprocess the MATH dataset for GRPO training.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split ("train" or "test").
        system_prompt: System message content prepended to every prompt.
        seed: Random seed for shuffling.
        max_level: Keep only problems with level <= max_level (1-5).
                   Lower values give denser reward signal for small models.

    Returns:
        HuggingFace Dataset with columns: prompt, answer, level, type.
    """
    logger.info("Loading dataset %s split=%s ...", dataset_name, split)
    raw = load_dataset(dataset_name, split=split)
    logger.info("Loaded %d samples. Preprocessing...", len(raw))

    if max_level < 5:
        keep = {f"Level {i}" for i in range(1, max_level + 1)}
        raw = raw.filter(lambda ex: ex["level"] in keep)
        logger.info("Filtered to levels 1-%d: %d samples remaining", max_level, len(raw))

    system_prompt = system_prompt.strip()

    def preprocess(example: dict) -> dict:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
        answer = _extract_last_boxed(example["solution"])
        return {
            "prompt": prompt,
            "answer": answer,
            "level": example["level"],
            "type": example["type"],
        }

    processed = raw.map(
        preprocess,
        remove_columns=raw.column_names,
        desc=f"Preprocessing {split} split",
    )
    before = len(processed)
    processed = processed.filter(lambda ex: bool(ex["answer"]))
    dropped = before - len(processed)
    if dropped:
        logger.warning("Dropped %d examples with empty gold answers (no \\boxed{} in solution).", dropped)
    processed = processed.shuffle(seed=seed)
    logger.info(
        "Dataset ready: %d samples, columns=%s", len(processed), processed.column_names
    )
    return processed
