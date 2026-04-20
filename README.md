# AutoLab

Auto Lab is a control plane that lets a AI agent operate RL training
runs — starting, forking, modifying, killing, and evaluating GRPO experiments — through a
structured 16-tool API backed by a trainer pool, telemetry layer, and persistent experiment
store. 

[https://aritra.io/autolab](https://aritra.io/autolab)

## Architecture

| Component | File | Role |
|---|---|---|
| **ControlPlane** | `autolab/control_plane/plane.py` | 16-tool agent API — write (start_run, fork, kill, modify, eval, set_active, set_cadence), read (get_run_details, list_runs, get_history, get_sample, compute_trend, get_config, get_eval), session (sleep, finalize) |
| **TrainerPool** | `autolab/trainer_pool/pool.py` | GRPOTrainer lifecycle state machines — atomic fork with checkpoint copy and rollback |
| **TelemetryLayer** | `autolab/telemetry/layer.py` | Queue drain → sparse ObservationEvents; windowed trends (slope, mean, std) computed in-layer at emission time |
| **MetadataStore** | `autolab/store/metadata_store.py` | SQLite — configs, runs, observations, samples, modifications, evals, sessions; queryable across sessions |
| **AgentLoop** | `autolab/agent/loop.py` | LLM loop, tool dispatch, budget enforcement; sleep/wake with four interrupt sources (timer, anomaly, eval complete, human message) |

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
run_artifacts/            Phase c-015 session artifacts (DB, logs, conversation)
```

## Known limitations

Fork overrides replace the entire `grpo` config block rather than deep-merging; the next
iteration will deep-merge fork overrides and accept per-call eval generation configs.

## License

```
MIT License — Copyright (c) 2026 Aritra Ghosh
```
