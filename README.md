# AutoLab

AutoLab is a control plane that lets a language model agent operate machine learning training
runs — starting, forking, modifying, killing, and evaluating GRPO experiments — through a
structured 16-tool API backed by a trainer pool, telemetry layer, and persistent experiment
store. Full write-up at aritra.io/autolab. This is a research artifact, not a production system.

## Architecture

Five components connect end-to-end. The **ControlPlane** (`autolab/control_plane/plane.py`)
exposes 16 typed tools the agent calls — write operations (start_run, fork, kill, modify, eval,
set_active, set_cadence), read operations (get_run_details, list_runs, get_history, get_sample,
compute_trend, get_config, get_eval), and session ops (sleep, finalize). The **TrainerPool**
(`autolab/trainer_pool/pool.py`) manages GRPOTrainer instances as lifecycle state machines,
supporting atomic fork with checkpoint copy and rollback. The **TelemetryLayer**
(`autolab/telemetry/layer.py`) drains step metrics from a queue, computes windowed trends
(slope, mean, std) in-layer at emission time, and writes sparse ObservationEvents to the store
and JSONL logs. The **MetadataStore** (`autolab/store/metadata_store.py`) persists all state in
SQLite — configs, runs, observations, samples, modifications, evals, and sessions — queryable
across sessions. The **AgentLoop** (`autolab/agent/loop.py`) runs the LLM, dispatches tool
calls, enforces the compute budget, and manages the sleep/wake cycle with four interrupt sources:
timer expiry, hard anomaly, eval completion, and human message.

## Demo

Phase c-015 (April 18 2026): a six-hour session in which the agent exercised every tool in the
surface — start, fork, modify, kill, eval, finalize — and diagnosed two interface bugs from tool
output alone. Full narrative in the writeup. Session artifacts are in `run_artifacts/`. To
build the replay JSON:

```bash
python scripts/build_replay.py \
  --session-id phase-c-015 \
  --db run_artifacts/store/autolab-c015.db \
  --logs-dir run_artifacts/logs/ \
  --sessions-dir run_artifacts/sessions/
```

The output is a structured JSON document containing every agent decision, tool call,
observation, and diff — the full session in one file.

## How to run

Requires a GPU (A100 tested) and `ANTHROPIC_API_KEY`.

```bash
pip install -e .

# On RunPod: apply TRL 0.19.1 + transformers 5.x compatibility patches once per instance
python scripts/patch_trl.py

# Run an agent session
python scripts/run_session.py --session-config autolab/configs/session_c015.yaml
```

Session artifacts write to `sessions/{session_id}/`, training logs to `logs/`, store to
`store/autolab.db`.

## Repository structure

```
autolab/
  agent/          AgentLoop, SessionRunner, budget, sleep, history, system prompt
  control_plane/  ControlPlane (16 tools), specs, types
  trainer_pool/   TrainerPool, ManagedRun, callbacks (PauseCallback, HotModifyCallback)
  training/       GRPOTrainer factory, dataset, rewards, logging callback
  telemetry/      TelemetryLayer, ObservationEvent, trend computation
  store/          MetadataStore (SQLite), JSONL event log
  eval/           Evaluator (held-out MATH accuracy)
  configs/        Training and session YAML configs
scripts/
  run_session.py          Agent session entry point
  run_training.py         Direct training (no agent)
  patch_trl.py            TRL 0.19.1 + transformers 5.x compatibility patches
  build_replay.py         Build session replay JSON from artifacts
  integration_test_cp.py  8-step GPU integration test
  eval.py                 Standalone checkpoint evaluation
tests/
  test_agent_loop.py      Agent loop smoke tests (mock LLM, no GPU)
  test_reward_smoke.py    Reward function unit tests
run_artifacts/         Phase c-015 session artifacts (DB, logs, conversation)
```

## Known limitations

Fork overrides replace the entire `grpo` config block rather than deep-merging; the next
iteration will deep-merge fork overrides and accept per-call eval generation configs.

## License

MIT. See LICENSE.
