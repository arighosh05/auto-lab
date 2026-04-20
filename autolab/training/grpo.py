"""
GRPOTrainer factory.

build_trainer() is a pure function: given a config dict (parsed from YAML),
it returns a fully constructed GRPOTrainer ready to call .train() on.

Design:
  - No side effects beyond constructing the trainer object
  - All hyperparameters come from config_dict; nothing hardcoded
  - Model is passed as a string ID so GRPOTrainer handles device placement
    and creates the reference model clone internally
  - Easy to swap model, dataset, or reward functions for later phases

GRPOConfig field filtering:
  GRPOConfig inherits from TrainingArguments (frozen dataclass). Passing
  unknown keys raises TypeError. We use dataclasses.fields() to filter
  the YAML dict to only valid GRPOConfig field names at runtime, so the
  code stays forward-compatible as TRL adds or removes config fields.
"""

import dataclasses
import logging

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from autolab.training.callbacks import LoggingCallback
from autolab.training.data import build_dataset
from autolab.training.rewards import accuracy_reward, format_reward

logger = logging.getLogger(__name__)


def _build_grpo_config(grpo_cfg: dict) -> GRPOConfig:
    """Build a GRPOConfig from a dictionary, ignoring unknown keys.

    Args:
        grpo_cfg: Dict from the YAML's 'grpo' section, optionally augmented
                  with 'model_init_kwargs'.

    Returns:
        GRPOConfig instance with all recognised fields set.
    """
    valid_fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    filtered = {k: v for k, v in grpo_cfg.items() if k in valid_fields}

    unknown = set(grpo_cfg.keys()) - valid_fields
    if unknown:
        logger.warning("Ignoring unknown GRPOConfig keys: %s", sorted(unknown))

    return GRPOConfig(**filtered)


def build_trainer(config_dict: dict, reward_funcs: list = None) -> GRPOTrainer:
    """Construct a GRPOTrainer from a parsed YAML config dict.

    Expected top-level keys in config_dict:
        model_name (str)            — HuggingFace model ID or local path
        model_revision (str)        — git revision, default "main"
        dataset_name (str)          — HuggingFace dataset ID
        dataset_split_train (str)   — split name, default "train"
        dataset_seed (int)          — shuffle seed, default 42
        system_prompt (str)         — system message for every sample
        grpo (dict)                 — GRPOConfig kwargs
        model_init_kwargs (dict)    — passed to GRPOConfig.model_init_kwargs

    Args:
        config_dict: Full parsed YAML config.
        reward_funcs: Optional list of reward functions. If None, uses the
            default [accuracy_reward, format_reward]. Pass pre-wrapped functions
            here to capture samples without modifying the training loop.

    Returns:
        GRPOTrainer ready to call .train().
    """
    model_name: str = config_dict["model_name"]
    model_revision: str = config_dict.get("model_revision", "main")
    system_prompt: str = config_dict["system_prompt"]
    dataset_name: str = config_dict["dataset_name"]
    split: str = config_dict.get("dataset_split_train", "train")
    dataset_seed: int = config_dict.get("dataset_seed", 42)
    dataset_max_level: int = config_dict.get("dataset_max_level", 5)
    model_init_kwargs: dict = dict(config_dict.get("model_init_kwargs", {}))
    # Normalise torch_dtype: agent may write "torch.bfloat16"; strip the "torch." prefix.
    if "torch_dtype" in model_init_kwargs:
        v = model_init_kwargs["torch_dtype"]
        if isinstance(v, str) and v.startswith("torch."):
            model_init_kwargs["torch_dtype"] = v[len("torch."):]

    # Merge model_init_kwargs into grpo dict so GRPOConfig knows how to load
    # the model (dtype, attention implementation, etc.)
    grpo_raw = dict(config_dict.get("grpo", {}))
    grpo_raw["model_init_kwargs"] = model_init_kwargs

    grpo_config = _build_grpo_config(grpo_raw)

    # Tokenizer — Qwen3 requires explicit pad_token and left-padding for batch gen
    logger.info("Loading tokenizer: %s (revision=%s)", model_name, model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%r)", tokenizer.eos_token)

    # Qwen3 thinking mode: enable_thinking=True lets the model reason via <think> tags
    # before writing its final answer.  enable_thinking=False pre-fills <think>\n\n</think>
    # so the model skips reasoning and writes a concise answer directly (~50-200 tokens).
    # The reward functions strip <think> blocks before verification either way.
    enable_thinking: bool = config_dict.get("enable_thinking", False)
    _orig_act = tokenizer.apply_chat_template

    def _patched_act(conversation, **kwargs):
        kwargs.setdefault("enable_thinking", enable_thinking)
        return _orig_act(conversation, **kwargs)

    tokenizer.apply_chat_template = _patched_act
    logger.info("Qwen3: applied enable_thinking=%s patch to apply_chat_template", enable_thinking)

    # Dataset
    dataset = build_dataset(
        dataset_name=dataset_name,
        split=split,
        system_prompt=system_prompt,
        seed=dataset_seed,
        max_level=dataset_max_level,
    )

    # Reward functions — order determines metric key names and reward_weights order.
    # TRL logs as "rewards/<fn.__name__>": accuracy_reward → rewards/accuracy_reward
    if reward_funcs is None:
        reward_funcs = [accuracy_reward, format_reward]

    logger.info(
        "Building GRPOTrainer: model=%s  dataset_size=%d  num_generations=%d",
        model_name,
        len(dataset),
        grpo_config.num_generations,
    )

    trainer = GRPOTrainer(
        model=model_name,           # String: trainer handles device placement + ref clone
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[LoggingCallback()],
    )

    return trainer
