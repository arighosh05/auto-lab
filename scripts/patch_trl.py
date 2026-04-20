#!/usr/bin/env python3
"""Apply TRL 0.19.1 compatibility patches for transformers 5.x.

Run once per fresh RunPod instance. Idempotent — safe to run multiple times.

Patches:
  1. _is_package_available() returns (bool, version) tuple in transformers 5.x;
     TRL expects just a bool. Wrap it.
  2. model.warnings_issued["estimate_tokens"] removed from GenerationMixin in
     transformers 5.x. Guard with hasattr.
  3. unwrap_model_for_generation() never switches to eval() mode. With
     gradient_checkpointing=True, Qwen3DecoderLayer sets past_key_values=None
     in train() mode, breaking autoregressive generation (garbage outputs, zero
     reward). Fix: eval() before yield, train() in finally.
"""

import re
import sys
from pathlib import Path

TRL_ROOT = Path("/usr/local/lib/python3.11/dist-packages/trl")

_PATCH1_MARKER = "# autolab_patch: _is_package_available tuple wrapper"
_PATCH2_MARKER = "# autolab_patch: warnings_issued guard"
_PATCH3_MARKER = "# autolab_patch: eval/train toggle for generation"


def patch_import_utils() -> None:
    path = TRL_ROOT / "import_utils.py"
    src = path.read_text()

    if _PATCH1_MARKER in src:
        print("Patch 1 (import_utils): already applied.")
        return

    old = "from transformers.utils.import_utils import _is_package_available"
    if old not in src:
        print("Patch 1: target string not found — TRL version may differ. Skipping.")
        return

    new = (
        f"{_PATCH1_MARKER}\n"
        "from transformers.utils.import_utils import _is_package_available as _ipa_raw\n"
        "def _is_package_available(pkg, return_version=False):\n"
        "    r = _ipa_raw(pkg)\n"
        "    is_avail = bool(r[0]) if isinstance(r, tuple) else bool(r)\n"
        "    version = r[1] if isinstance(r, tuple) else None\n"
        "    if return_version:\n"
        "        return is_avail, version\n"
        "    return is_avail\n"
    )
    path.write_text(src.replace(old, new))
    print("Patch 1 (import_utils): applied.")


def patch_grpo_trainer() -> None:
    path = TRL_ROOT / "trainer" / "grpo_trainer.py"
    src = path.read_text()

    if _PATCH2_MARKER in src:
        print("Patch 2 (grpo_trainer): already applied.")
        return

    # Capture leading whitespace so the if-body is indented correctly.
    m = re.search(r'( *)model\.warnings_issued\["estimate_tokens"\] = True', src)
    if not m:
        print("Patch 2: target string not found — TRL version may differ. Skipping.")
        return

    indent = m.group(1)
    old = m.group(0)
    new = (
        f'{indent}if hasattr(model, "warnings_issued"):  {_PATCH2_MARKER}\n'
        f'{indent}    model.warnings_issued["estimate_tokens"] = True'
    )
    path.write_text(src.replace(old, new, 1))
    print("Patch 2 (grpo_trainer): applied.")


def patch_unwrap_model_for_generation() -> None:
    path = TRL_ROOT / "models" / "utils.py"
    src = path.read_text()

    if _PATCH3_MARKER in src:
        print("Patch 3 (unwrap_model_for_generation): already applied.")
        return

    old = "    else:\n        yield unwrapped_model"
    if old not in src:
        print("Patch 3: target string not found — TRL version may differ. Skipping.")
        return

    new = (
        f"    else:  {_PATCH3_MARKER}\n"
        "        unwrapped_model.eval()\n"
        "        try:\n"
        "            yield unwrapped_model\n"
        "        finally:\n"
        "            unwrapped_model.train()"
    )
    path.write_text(src.replace(old, new, 1))
    print("Patch 3 (unwrap_model_for_generation): applied.")


if __name__ == "__main__":
    if not TRL_ROOT.exists():
        print(f"TRL not found at {TRL_ROOT}. Exiting.", file=sys.stderr)
        sys.exit(1)
    patch_import_utils()
    patch_grpo_trainer()
    patch_unwrap_model_for_generation()
    print("Done.")
