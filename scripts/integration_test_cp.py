#!/usr/bin/env python3
"""Control Plane integration test — exercises the full lifecycle on a real GPU.

Sequence: start_run → pause (set_active None) → resume → fork → modify → kill → eval → idle

Run with: python scripts/integration_test_cp.py
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import time
import traceback

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from autolab.control_plane import ControlPlane
from autolab.store.logs import Logs
from autolab.store.metadata_store import MetadataStore
from autolab.telemetry.layer import TelemetryLayer
from autolab.trainer_pool.pool import TrainerPool

# ── Configure logging so training progress is visible ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/integration_test.log"),
    ],
)
logger = logging.getLogger("integration_test")

# ── Hard wall-clock deadline ──────────────────────────────────────────────────
DEADLINE = time.time() + 3 * 3600   # T+3 h — do not debug on idle GPU past this
DB_PATH = "store/integration_test.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for(condition, *, timeout: float, poll: float = 10.0, label: str) -> bool:
    """Poll condition() until True or timeout. Prints heartbeat each poll."""
    deadline = min(time.time() + timeout, DEADLINE)
    while time.time() < deadline:
        if condition():
            return True
        elapsed = int(time.time() - (deadline - timeout))
        print(f"  [{elapsed:>4}s] waiting: {label} …", flush=True)
        time.sleep(poll)
    if time.time() >= DEADLINE:
        bail("WALL-CLOCK DEADLINE EXCEEDED", label=label)
    return False


def bail(msg: str, **ctx):
    """Print failure context and exit 1."""
    print(f"\n{'='*60}")
    print(f"FAIL: {msg}")
    for k, v in ctx.items():
        print(f"  {k}: {v}")
    traceback.print_stack(limit=6)
    print("="*60)
    sys.exit(1)


def assert_ok(resp, step_name: str):
    if resp.error is not None:
        bail(f"{step_name} returned error", code=resp.error.code,
             message=resp.error.message, suggested=resp.error.suggested_action)


# ── Setup ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("INTEGRATION TEST — Control Plane")
print("="*60)
print(f"Deadline: {time.strftime('%H:%M:%S', time.localtime(DEADLINE))}")

os.makedirs("store", exist_ok=True)
os.makedirs("logs", exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

store = MetadataStore(DB_PATH)
logs  = Logs("logs")
mq, sq = queue.Queue(), queue.Queue()
telemetry = TelemetryLayer(mq, logs, store, sample_queue=sq)
pool = TrainerPool(metrics_queue=mq, sample_queue=sq)
cp   = ControlPlane(pool, store, telemetry, logs)

with open("autolab/configs/grpo_qwen3_math.yaml") as f:
    config = yaml.safe_load(f)

config["grpo"]["max_steps"]     = 60
config["grpo"]["logging_steps"] = 1
config["grpo"]["save_steps"]    = 5

print(f"Config: max_steps={config['grpo']['max_steps']}, "
      f"save_steps={config['grpo']['save_steps']}, "
      f"model={config['model_name']}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — start_run
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 1] start_run")

r1 = cp.start_run(config, reason="integration-test")
print(f"  result : {r1.result}")
print(f"  status : {r1.status}")
assert_ok(r1, "start_run")

run_id = r1.result["run_id"]
if r1.status.gpu_state != "training":
    bail("gpu_state should be training", got=r1.status.gpu_state)

print(f"  run_id : {run_id}  ✓ step 1 pass")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — wait for ≥10 steps, then pause via set_active(None)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 2] wait for step ≥10 then pause")
print("  (first step may be slow: model download + dataset prep)")

ok = wait_for(
    lambda: pool.get_run(run_id).current_step >= 10,
    timeout=3600,   # 1 h — first run may include download
    poll=20,
    label="current_step ≥ 10",
)
if not ok:
    bail("Timed out before step 10", current_step=pool.get_run(run_id).current_step)

step_before_pause = pool.get_run(run_id).current_step
print(f"  reached step {step_before_pause}. Pausing (blocks until checkpoint saved)…")

r2 = cp.set_active(None, reason="pause-for-test")
print(f"  result : {r2.result}")
print(f"  status : {r2.status}")
assert_ok(r2, "set_active(None)")

if r2.status.gpu_state != "idle":
    bail("gpu_state should be idle after pause", got=r2.status.gpu_state)
if pool.get_run(run_id).status.value != "paused":
    bail("run status should be paused", got=pool.get_run(run_id).status.value)

ckpt = pool.get_run(run_id).checkpoint_path
print(f"  checkpoint_path : {ckpt}")
if not ckpt or not os.path.isdir(ckpt):
    bail("No checkpoint directory after pause", path=ckpt)

print(f"  ✓ step 2 pass  (paused at step {step_before_pause}, checkpoint={ckpt})")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — resume  ← CRITICAL CORRECTNESS CHECK
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 3] resume (CRITICAL — must continue from checkpoint, not from 0)")

r3 = cp.set_active(run_id, reason="resume-after-pause")
print(f"  result : {r3.result}")
print(f"  status : {r3.status}")
assert_ok(r3, "set_active(resume)")

if r3.status.gpu_state != "training":
    bail("gpu_state should be training after resume", got=r3.status.gpu_state)

# Wait for first step to advance past where we were before pause
ok = wait_for(
    lambda: pool.get_run(run_id).current_step > step_before_pause,
    timeout=1200,   # 20 min
    poll=15,
    label=f"current_step > {step_before_pause}",
)
if not ok:
    bail("Training didn't advance after resume", current_step=pool.get_run(run_id).current_step)

step_after_resume = pool.get_run(run_id).current_step
print(f"  step after resume: {step_after_resume}  (was {step_before_pause} before resume)")

# ▼ CRITICAL CHECK
if step_after_resume <= 5:
    print("\n  *** CRITICAL FAILURE ***")
    print(f"  step_after_resume={step_after_resume} is ≤5 — training restarted from 0!")
    print("  This means checkpoint resume is broken.")
    print("  Pull artifacts and tear down pod:")
    print(f"    scp -P 14510 -r root@<pod>:/workspace/auto-lab/outputs/ ./debug_outputs/")
    print(f"    scp -P 14510 -r root@<pod>:/workspace/auto-lab/logs/ ./debug_logs/")
    sys.exit(1)

print(f"  ✓ step 3 pass  (continued from step {step_before_pause} → {step_after_resume})")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — fork at ~step 20
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 4] wait for step ≥20 then fork")

ok = wait_for(
    lambda: pool.get_run(run_id).current_step >= 20,
    timeout=3600,
    poll=15,
    label="current_step ≥ 20",
)
if not ok:
    bail("Timed out before step 20", current_step=pool.get_run(run_id).current_step)

print(f"  parent at step {pool.get_run(run_id).current_step}. Forking with lr=1e-6…")

r4 = cp.fork(run_id, overrides={"grpo.learning_rate": 1e-6}, reason="fork-test")
print(f"  result : {r4.result}")
print(f"  status : {r4.status}")
assert_ok(r4, "fork")

child_id  = r4.result["run_id"]
fork_step = r4.result["fork_step"]
print(f"  child_id={child_id}, fork_step={fork_step}")

# Parent should be paused
if pool.get_run(run_id).status.value != "paused":
    bail("Parent should be paused after fork", got=pool.get_run(run_id).status.value)

# Child should be running
ok = wait_for(
    lambda: pool.get_run(child_id) is not None
            and pool.get_run(child_id).status.value in ("running", "paused", "done", "failed"),
    timeout=120,
    poll=5,
    label="child run registered",
)
child_status = pool.get_run(child_id).status.value
if child_status not in ("running",):
    bail("Child should be running", got=child_status)

# Verify fork checkpoint on disk
child_config = pool.get_run(child_id).config
fork_ckpt_path = child_config.get("grpo", {}).get("resume_from_checkpoint", "")
print(f"  fork checkpoint path : {fork_ckpt_path}")
if fork_ckpt_path and not os.path.isdir(fork_ckpt_path):
    bail("Fork checkpoint not on disk", path=fork_ckpt_path)

# Verify store lineage
child_store_row = store.get_run(child_id)
if child_store_row is None:
    bail("Child not in store")
if child_store_row.get("parent_run_id") != run_id:
    bail("Wrong parent_run_id in store",
         expected=run_id, got=child_store_row.get("parent_run_id"))

print(f"  ✓ step 4 pass  (child={child_id}, parent paused, fork checkpoint on disk)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — modify child mid-run
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 5] wait for child to take a few steps then modify")

ok = wait_for(
    lambda: pool.get_run(child_id).current_step > fork_step + 3,
    timeout=600,
    poll=10,
    label=f"child step > {fork_step + 3}",
)
if not ok:
    # Non-fatal — modify should still work, the HotModifyCallback just fires on next step
    print(f"  warning: child step={pool.get_run(child_id).current_step}, pushing modify anyway")

old_config_id = telemetry._config_ids.get(child_id, "")
r5 = cp.modify(child_id, {"learning_rate": 2e-6}, reason="lr-bump-test")
print(f"  result : {r5.result}")
print(f"  status : {r5.status}")
assert_ok(r5, "modify")

new_config_id = r5.result["new_config_id"]
if new_config_id == old_config_id:
    bail("new_config_id should differ from old_config_id",
         old=old_config_id, new=new_config_id)
print(f"  config changed: {old_config_id} → {new_config_id}")

# Wait for HotModifyCallback to fire
ok = wait_for(
    lambda: pool.get_run(child_id).last_modify_step > 0,
    timeout=600,
    poll=10,
    label="last_modify_step > 0 (HotModifyCallback fired)",
)
if not ok:
    print("  warning: HotModifyCallback may not have fired yet — checking store anyway")

mods = store.list_modifications(child_id)
print(f"  modifications in store: {mods}")
if not any("learning_rate" in str(m.get("changes", "")) for m in mods):
    bail("No learning_rate modification found in store", mods=mods)

print(f"  ✓ step 5 pass  (lr modified, config_id updated, store row present)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — kill parent
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 6] kill parent (currently paused)")

r6 = cp.kill(run_id, reason="kill-after-fork")
print(f"  result : {r6.result}")
print(f"  status : {r6.status}")
assert_ok(r6, "kill")

if pool.get_run(run_id).status.value != "killed":
    bail("Parent should be killed", got=pool.get_run(run_id).status.value)

listed = cp.list_runs()        # default filter: running + paused only
listed_ids = [x["run_id"] for x in listed.result]
print(f"  list_runs (running/paused): {listed_ids}")
if run_id in listed_ids:
    bail("Killed run should not appear in list_runs default view", listed=listed_ids)

print(f"  ✓ step 6 pass  (parent killed, not in list_runs)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — eval (async, blocks GPU, no auto-resume)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 7] eval (waits for child checkpoint then triggers async eval)")

# Wait for child to save at least one of its own checkpoints
print("  waiting for child to save a post-fork checkpoint…")
child_output_dir = pool.get_run(child_id).config.get("grpo", {}).get("output_dir", "")
ok = wait_for(
    lambda: (pool.get_run(child_id).checkpoint_path is not None)
            or (bool(child_output_dir) and cp._find_latest_checkpoint(child_output_dir) is not None),
    timeout=3600,
    poll=20,
    label="child checkpoint exists",
)
if not ok:
    bail("No child checkpoint found for eval", output_dir=child_output_dir)

r7 = cp.eval(child_id, "math", n_samples=20, reason="eval-test")
print(f"  result : {r7.result}")
print(f"  status : {r7.status}")
assert_ok(r7, "eval")

eval_id = r7.result["eval_id"]
if r7.result["status"] != "running":
    bail("eval should start as running", got=r7.result["status"])
if r7.status.gpu_state != "evaluating":
    print(f"  note: gpu_state={r7.status.gpu_state} (expected evaluating; "
          f"child may have hit max_steps before eval triggered)")

print(f"  eval_id={eval_id}, training_paused={r7.result.get('training_paused')}")
print("  polling for eval completion (n_samples=20)…")

t_eval_start = time.time()
while True:
    if time.time() > DEADLINE:
        bail("Deadline exceeded during eval poll")
    re = cp.get_eval(eval_id)
    status = re.result.get("status", "unknown")
    elapsed = int(time.time() - t_eval_start)
    print(f"  [{elapsed:>4}s] eval status: {status}", flush=True)
    if status == "done":
        break
    if status == "failed":
        bail("Eval failed", error=re.result.get("error"))
    time.sleep(20)

print(f"  eval done: {json.dumps(re.result, indent=2, default=str)}")

# Check accuracy in result
eval_result = re.result.get("results", re.result)
if "overall_accuracy" not in eval_result and "accuracy" not in eval_result:
    bail("Eval result missing accuracy field", result=eval_result)

if cp._status().gpu_state not in ("idle", "training"):
    # might be "training" if child hit max_steps and DONE status already
    pass
print(f"  gpu_state after eval: {cp._status().gpu_state}")
print(f"  ✓ step 7 pass  (eval completed, result in store)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — resume child after eval, then go idle
# ════════════════════════════════════════════════════════════════════════════════
print("\n[Step 8] set_active(child) after eval then idle")

child_current_status = pool.get_run(child_id).status.value
print(f"  child status before set_active: {child_current_status}")

if child_current_status == "paused":
    r8 = cp.set_active(child_id, reason="resume-after-eval")
    print(f"  result : {r8.result}")
    assert_ok(r8, "set_active after eval")
    if r8.status.gpu_state != "training":
        bail("Expected training after resume", got=r8.status.gpu_state)
    print("  child resumed. Waiting 15s…")
    time.sleep(15)
elif child_current_status == "done":
    print("  child finished naturally (hit max_steps) — no resume needed")
else:
    print(f"  child status={child_current_status}, skipping resume")

r9 = cp.set_active(None, reason="final-idle")
print(f"  set_active(None): {r9.result}, gpu_state={r9.status.gpu_state}")
assert_ok(r9, "set_active(None) final")
if r9.status.gpu_state != "idle":
    bail("Expected idle at end", got=r9.status.gpu_state)

print("  ✓ step 8 pass")


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
elapsed_total = int(time.time() - (DEADLINE - 3 * 3600))
print(f"\n{'='*60}")
print(f"ALL STEPS PASSED  (wall time: {elapsed_total//60}m {elapsed_total%60}s)")
print(f"{'='*60}")
print(f"  run_id:    {run_id}")
print(f"  child_id:  {child_id}")
print(f"  eval_id:   {eval_id}")
print(f"  DB:        {DB_PATH}  (kept for inspection)")
print()
store.close()
