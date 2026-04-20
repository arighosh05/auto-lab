#!/usr/bin/env python3
"""
Training entry point for autolab Phase 1.

Launches a GRPO training run through the full telemetry stack:
  TrainerPool → MetricsHookCallback + reward wrappers
    → TelemetryLayer → Logs (JSONL) + MetadataStore (SQLite)

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --config autolab/configs/grpo_qwen3_math.yaml
    python scripts/run_training.py --config <path> --cadence 50 --store-dir store --logs-dir logs
"""

import argparse
import json
import logging
import queue
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolab.store.logs import Logs
from autolab.store.metadata_store import MetadataStore
from autolab.telemetry.layer import TelemetryLayer
from autolab.trainer_pool.pool import TrainerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GRPO fine-tuning (Phase 1).")
    p.add_argument(
        "--config",
        default="autolab/configs/grpo_qwen3_math.yaml",
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--cadence",
        type=int,
        default=50,
        help="Emit an observation event every N training steps (default: 50).",
    )
    p.add_argument(
        "--store-dir",
        default="store",
        help="Directory for SQLite database (default: store/).",
    )
    p.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for JSONL event logs (default: logs/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    logger.info("Loading config: %s", config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Ensure runtime directories exist
    store_dir = Path(args.store_dir)
    logs_dir = Path(args.logs_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate storage layer
    store = MetadataStore(db_path=store_dir / "autolab.db")
    logs = Logs(logs_dir=logs_dir)

    # Queues — unbounded so MetricsHookCallback never blocks training
    metrics_queue: queue.Queue = queue.Queue()
    sample_queue: queue.Queue = queue.Queue()

    # Telemetry layer wires queues to storage
    telemetry = TelemetryLayer(
        metrics_queue=metrics_queue,
        logs=logs,
        metadata_store=store,
        observation_cadence=args.cadence,
        sample_queue=sample_queue,
    )

    # Trainer pool owns the ManagedRun
    pool = TrainerPool(metrics_queue=metrics_queue, sample_queue=sample_queue)

    # Pre-register the run so config_id is known before training starts.
    # Pass the same run_id to pool.start_run() so everything is consistent.
    run_name = config.get("grpo", {}).get("run_name", "run")
    run_id = f"{run_name}-{int(time.time())}"

    telemetry.register_run(run_id, config)

    config_id = telemetry._config_ids[run_id]
    store.insert_config(config_id=config_id, config_json=json.dumps(config, default=str))
    store.insert_run(
        run_id=run_id,
        config_id=config_id,
        status="running",
        start_time=time.time(),
        creation_reason="initial run from CLI",
    )

    logger.info("Starting telemetry drain thread")
    telemetry.start()

    logger.info("Starting training run: %s", run_id)
    try:
        pool.start_run(config, run_id=run_id)  # BLOCKS until train() returns
    finally:
        # Always stop telemetry even if training fails
        telemetry.stop()

    # Update run record with final status
    run = pool.get_run(run_id)
    final_metrics_json = json.dumps(run.final_metrics, default=str) if run.final_metrics else "{}"
    store.update_run(
        run_id=run_id,
        status=run.status.value,
        end_time=run.end_time or time.time(),
        final_metrics_json=final_metrics_json,
    )

    # Summary
    history = store.get_history(run_id)
    n_obs = len(history)
    n_samples = len(store.get_samples(run_id, limit=1_000_000))
    log_path = logs_dir / f"{run_id}.jsonl"
    db_path = store_dir / "autolab.db"

    print("\n=== Training Complete ===")
    print(f"run_id      : {run_id}")
    print(f"status      : {run.status.value}")
    print(f"observations: {n_obs}")
    print(f"samples     : {n_samples}")
    print(f"log file    : {log_path}")
    print(f"database    : {db_path}")
    if run.final_metrics:
        relevant = {
            k: v for k, v in run.final_metrics.items()
            if k in ("loss", "reward", "rewards/accuracy_reward/mean", "rewards/format_reward/mean")
        }
        if relevant:
            print(f"final       : {relevant}")

    store.close()

    if run.error:
        logger.error("Run finished with error: %s", run.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
