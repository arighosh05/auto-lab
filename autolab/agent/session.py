"""SessionRunner — wires up all components and runs a training session."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "session_default.yaml"


def _load_config(session_config_path: str | Path) -> dict:
    """Load session config, merging with session_default.yaml defaults."""
    with _DEFAULT_CONFIG_PATH.open(encoding="utf-8") as f:
        defaults = yaml.safe_load(f)

    with Path(session_config_path).open(encoding="utf-8") as f:
        overrides = yaml.safe_load(f)

    merged = {**defaults, **(overrides or {})}
    return merged


class SessionRunner:
    """Orchestrates a single autolab agent session.

    Instantiates all infrastructure components, registers the session in the DB,
    runs the AgentLoop, and cleans up on exit.

    Args:
        config: Merged session config dict (from session YAML + defaults).
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def run(self) -> str:
        """Run the session. Returns the terminal status string."""
        import anthropic

        from autolab.agent.budget import BudgetTracker
        from autolab.agent.history import ConversationHistory
        from autolab.agent.human_queue import HumanMessageQueue
        from autolab.agent.loop import AgentLoop
        from autolab.agent.prompt import render_system_prompt
        from autolab.agent.sleep import SleepScheduler
        from autolab.control_plane.plane import ControlPlane
        from autolab.store.logs import Logs
        from autolab.store.metadata_store import MetadataStore
        from autolab.telemetry.layer import TelemetryLayer
        from autolab.trainer_pool.pool import TrainerPool

        cfg = self._config
        session_id: str = cfg["session_id"]
        session_dir = Path("sessions") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SessionRunner: starting session %s", session_id)

        # Infrastructure
        store = MetadataStore(cfg.get("db_path", "store/autolab.db"))
        logs = Logs(cfg.get("logs_dir", "logs"))
        mq: queue.Queue = queue.Queue()
        sq: queue.Queue = queue.Queue()
        telemetry = TelemetryLayer(mq, logs, store, sample_queue=sq)
        pool = TrainerPool(metrics_queue=mq, sample_queue=sq)
        cp = ControlPlane(pool, store, telemetry, logs)
        telemetry.start()

        # Load initial training config
        initial_config_path = cfg.get("initial_config_path")
        if initial_config_path:
            with Path(initial_config_path).open(encoding="utf-8") as f:
                initial_config = yaml.safe_load(f)
        else:
            initial_config = {}

        # Agent components
        budget = BudgetTracker(float(cfg["budget_seconds"]))
        human_queue = HumanMessageQueue(session_dir / "inbox.txt")
        sleep_scheduler = SleepScheduler(logs, cp)
        history = ConversationHistory(
            max_live_turns=int(cfg.get("history_keep_last_n", 10)),
            summarizer_model=cfg.get("history_summarizer_model", "claude-haiku-4-5-20251001"),
        )
        system_prompt = render_system_prompt(cp, initial_config)
        (session_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        stop_event = threading.Event()

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        loop = AgentLoop(
            cp=cp,
            store=store,
            session_id=session_id,
            session_dir=session_dir,
            goal=cfg["goal"],
            success_criterion=cfg["success_criterion"],
            budget=budget,
            human_queue=human_queue,
            sleep_scheduler=sleep_scheduler,
            history=history,
            system_prompt=system_prompt,
            client=client,
            stop_event=stop_event,
            model=cfg.get("model_name", "claude-opus-4-6"),
            max_tokens=int(cfg.get("max_tokens_per_turn", 4096)),
            sleep_default=float(cfg.get("sleep_default_seconds", 300)),
        )

        # Register session in DB
        try:
            store.insert_session(
                session_id=session_id,
                started_at=time.time(),
                goal=cfg["goal"],
                success_criterion=cfg["success_criterion"],
                budget_seconds=float(cfg["budget_seconds"]),
            )
        except Exception:
            logger.exception("SessionRunner: failed to insert session row")

        terminal_status = "failed"
        try:
            terminal_status = loop.run()
            logger.info("SessionRunner: loop exited with status=%s", terminal_status)
        except Exception:
            logger.exception("SessionRunner: AgentLoop raised an unhandled exception")
            try:
                store.update_session(session_id, ended_at=time.time(), status="failed")
            except Exception:
                pass
        finally:
            sleep_scheduler.close()
            telemetry.stop()
            store.close()
            logger.info("SessionRunner: session %s done (status=%s)", session_id, terminal_status)

        return terminal_status
