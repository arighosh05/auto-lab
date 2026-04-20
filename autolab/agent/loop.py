"""AgentLoop — outer LLM loop that drives the autonomous training agent."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from autolab.control_plane.types import ErrorCode, StatusBar, ToolError, ToolResponse

logger = logging.getLogger(__name__)

# Mutating tools that are subject to budget enforcement.
_MUTATING_TOOLS = frozenset(
    {"start_run", "fork", "modify", "set_active", "set_cadence", "eval"}
)

# Auto-finalize after this many consecutive turns with exhausted budget.
_MAX_OVERBUDGET_TURNS = 3


class AgentLoop:
    """Drives the LLM ↔ ControlPlane interaction loop for one session.

    This class is responsible for:
    - Invoking the LLM with the current conversation history
    - Routing tool calls to ControlPlane (or handling sleep/finalize directly)
    - Injecting budget info into every StatusBar
    - Enforcing budget limits on mutating tools
    - Writing every message to conversation.jsonl
    - Auto-finalizing on hard termination conditions

    All components are injected — the loop has no knowledge of how to construct them.

    Args:
        cp: ControlPlane instance.
        store: MetadataStore instance.
        session_id: Unique session identifier.
        session_dir: Path to sessions/{session_id}/.
        goal: Human-provided goal string.
        success_criterion: Human-provided success criterion string.
        budget: BudgetTracker instance.
        human_queue: HumanMessageQueue instance.
        sleep_scheduler: SleepScheduler instance.
        history: ConversationHistory instance.
        system_prompt: Pre-rendered system prompt string.
        client: Anthropic client (or mock with the same interface).
        stop_event: threading.Event set when .stop file detected.
        model: LLM model ID.
        max_tokens: Max tokens per LLM turn.
        sleep_default: Default sleep seconds when the tool omits the field.
    """

    def __init__(
        self,
        *,
        cp: Any,
        store: Any,
        session_id: str,
        session_dir: Path,
        goal: str,
        success_criterion: str,
        budget: Any,
        human_queue: Any,
        sleep_scheduler: Any,
        history: Any,
        system_prompt: str,
        client: Any,
        stop_event: threading.Event,
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        sleep_default: float = 300.0,
    ) -> None:
        self._cp = cp
        self._store = store
        self._session_id = session_id
        self._session_dir = Path(session_dir)
        self._goal = goal
        self._success_criterion = success_criterion
        self._budget = budget
        self._human_queue = human_queue
        self._sleep_scheduler = sleep_scheduler
        self._history = history
        self._system_prompt = system_prompt
        self._client = client
        self._stop_event = stop_event
        self._model = model
        self._max_tokens = max_tokens
        self._sleep_default = sleep_default

        self._finalized = False
        self._terminal_status = "failed"
        self._interrupt_requested = False
        self._interrupt_message_sent = False
        self._overbudget_turns = 0

        self._conv_log_path = self._session_dir / "conversation.jsonl"

    # =========================================================================
    # Public entry point
    # =========================================================================

    def run(self) -> str:
        """Run the agent loop until termination. Returns the terminal status string.

        Terminal statuses: 'finalized', 'human_interrupted', 'budget_exhausted', 'failed'.
        """
        # Build the initial user message
        initial_msg = self._format_initial_prompt()
        self._history.append_user(initial_msg)
        self._log_message({"role": "user", "content": initial_msg, "ts": time.time()})

        try:
            while not self._finalized:
                self._run_one_turn()
        except Exception:
            logger.exception("AgentLoop: unhandled exception — auto-finalizing as 'failed'")
            try:
                self._auto_finalize("failed")
            except Exception:
                logger.exception("AgentLoop: auto_finalize also failed")
            self._terminal_status = "failed"

        return self._terminal_status

    # =========================================================================
    # One turn
    # =========================================================================

    def _run_one_turn(self) -> None:
        # 1. Check .stop file
        if (self._session_dir / ".stop").exists():
            if not self._interrupt_requested:
                logger.info("AgentLoop: .stop file detected — setting interrupt flag")
            self._stop_event.set()
            self._interrupt_requested = True

        # 2. Drain human queue
        self._human_queue.poll()
        for msg_text in self._human_queue.drain():
            logger.info("AgentLoop: human message received: %s", msg_text[:100])
            self._history.append_user(msg_text)
            self._log_message({"role": "user", "content": msg_text, "ts": time.time()})

        # 3. Inject interrupt notice (once)
        if self._interrupt_requested and not self._interrupt_message_sent:
            notice = (
                "Human requested session end. Please call finalize() with your best "
                "current run ID and a brief summary of what was accomplished."
            )
            self._history.append_user(notice)
            self._log_message({"role": "user", "content": notice, "ts": time.time()})
            self._interrupt_message_sent = True

        # 4. Compress history if needed
        self._history.maybe_compress(self._client)

        # 5. LLM call
        logger.debug("AgentLoop: invoking LLM (model=%s)", self._model)
        response = self._client.messages.create(
            model=self._model,
            system=self._system_prompt,
            messages=_sanitize_messages(self._history.for_llm()),
            tools=self._cp.tool_specs(),
            max_tokens=self._max_tokens,
        )
        self._history.append_assistant(response)
        self._log_message({"role": "assistant", "content": _serialize_content(response.content), "ts": time.time()})

        # 6. Execute tool calls
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type != "tool_use":
                continue

            tool_name = block.name
            tool_args = dict(block.input)
            tool_id = block.id

            logger.info("AgentLoop: tool call %s(%s)", tool_name, list(tool_args.keys()))
            result = self._invoke_tool(tool_name, tool_args)
            result_str = json.dumps(result, default=str)

            self._history.append_tool_result(tool_id, result_str)
            self._log_message({
                "role": "tool_result",
                "tool_use_id": tool_id,
                "tool_name": tool_name,
                "content": result,
                "ts": time.time(),
            })

            if self._finalized:
                return

        # 7. Auto-finalize on hard conditions
        if self._interrupt_requested and not self._finalized:
            logger.info("AgentLoop: interrupt requested and agent did not finalize — auto-finalizing")
            self._auto_finalize("human_interrupted")
            return

        if self._budget.exhausted():
            self._overbudget_turns += 1
            logger.info(
                "AgentLoop: budget exhausted, overbudget turn %d/%d",
                self._overbudget_turns, _MAX_OVERBUDGET_TURNS,
            )
            if self._overbudget_turns >= _MAX_OVERBUDGET_TURNS:
                logger.info("AgentLoop: max overbudget turns reached — auto-finalizing")
                self._auto_finalize("budget_exhausted")

    # =========================================================================
    # Tool dispatch
    # =========================================================================

    def _invoke_tool(self, name: str, args: dict) -> dict:
        """Route a tool call, enforce budget, inject budget into status."""
        if name == "sleep":
            return self._handle_sleep(args)
        if name == "finalize":
            return self._handle_finalize(args)

        # Route to ControlPlane
        try:
            response: ToolResponse = self._cp.invoke_by_name(name, args)
        except ValueError as exc:
            # Unknown tool — return an error dict shaped like ToolResponse
            return {
                "status": dataclasses.asdict(self._cp._status()),
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": str(exc),
                    "retryable": False,
                    "suggested_action": "Check the tool name and try again.",
                },
            }

        # Inject budget into status
        response.status.budget_remaining_seconds = self._budget.remaining_seconds()

        # Budget enforcement for mutating tools
        if name in _MUTATING_TOOLS:
            if self._budget.exhausted():
                response.result = None
                response.error = ToolError(
                    code=ErrorCode.BUDGET_EXHAUSTED,
                    message="Compute budget exhausted. No more training operations allowed.",
                    retryable=False,
                    suggested_action=(
                        "Call finalize() with the best run ID found so far. "
                        "Reads, kill, sleep, and finalize still work."
                    ),
                )
            elif self._budget.near_exhausted():
                # Append warning to StatusBar.warnings (always visible; doesn't mutate result shape)
                if "budget 90% consumed" not in response.status.warnings:
                    response.status.warnings.append("budget 90% consumed")

        return dataclasses.asdict(response)

    def _handle_sleep(self, args: dict) -> dict:
        seconds = float(args.get("seconds", self._sleep_default))
        logger.info("AgentLoop: sleeping for up to %.0fs (reason: %s)", seconds, args.get("reason", ""))
        woken_by, actual = self._sleep_scheduler.wait(
            seconds, self._human_queue, self._stop_event
        )
        logger.info("AgentLoop: woke up (woken_by=%s, actual=%.1fs)", woken_by, actual)
        if woken_by == "interrupted":
            self._stop_event.set()
            self._interrupt_requested = True
        return {"woken_by": woken_by, "actual_seconds": round(actual, 1)}

    def _handle_finalize(self, args: dict) -> dict:
        winning_run_id = args.get("winning_run_id")
        summary = args.get("summary", "")
        logger.info(
            "AgentLoop: finalize called (winning_run_id=%s)", winning_run_id
        )
        self._store.update_session(
            self._session_id,
            ended_at=time.time(),
            winning_run_id=winning_run_id,
            final_summary=summary,
            status="finalized",
        )
        self._finalized = True
        self._terminal_status = "finalized"
        return {"session_status": "finalized"}

    def _auto_finalize(self, status: str) -> None:
        """Select the best run and write the session record without agent involvement."""
        leader = self._find_leader_run_id()
        logger.info("AgentLoop: auto-finalizing with status=%s, leader=%s", status, leader)
        self._store.update_session(
            self._session_id,
            ended_at=time.time(),
            winning_run_id=leader,
            final_summary=f"Auto-finalized: {status}",
            status=status,
        )
        self._finalized = True
        self._terminal_status = status

    def _find_leader_run_id(self) -> Optional[str]:
        """Best run: highest eval accuracy → highest training accuracy → None."""
        try:
            evals = self._store.list_evals_all_runs()
            if evals:
                return evals[0]["run_id"]
        except Exception:
            logger.exception("AgentLoop: list_evals_all_runs failed")

        try:
            runs = self._store.list_runs()
            best_run_id, best_acc = None, -1.0
            for run in runs:
                if run.get("status") not in ("running", "paused", "done"):
                    continue
                hist = self._store.get_history(run["run_id"])
                if hist:
                    acc = hist[-1].get("metrics", {}).get(
                        "rewards/accuracy_reward/mean", 0.0
                    )
                    if acc > best_acc:
                        best_acc, best_run_id = acc, run["run_id"]
            return best_run_id
        except Exception:
            logger.exception("AgentLoop: leader fallback also failed")
            return None

    # =========================================================================
    # Helpers
    # =========================================================================

    def _format_initial_prompt(self) -> str:
        return (
            f"## Session start\n\n"
            f"**Goal:** {self._goal}\n\n"
            f"**Success criterion:** {self._success_criterion}\n\n"
            f"**Budget:** {self._budget.budget_seconds:.0f} seconds of wall-clock compute\n\n"
            f"The initial config is already in the system prompt under 'Example config'. "
            f"Start a run with it and begin optimizing."
        )

    def _log_message(self, message: dict) -> None:
        """Append a message to conversation.jsonl."""
        try:
            with self._conv_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(message, default=str) + "\n")
        except OSError:
            logger.exception("AgentLoop: failed to write conversation log")


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Round-trip messages through JSON to ensure all values are serializable.

    Converts any non-JSON-native objects (e.g. torch.dtype, numpy arrays) to
    their string representations via default=str.
    """
    return json.loads(json.dumps(messages, default=str))


def _serialize_content(content: Any) -> list[dict]:
    """Convert Anthropic content blocks to serializable dicts."""
    result = []
    if not isinstance(content, (list, tuple)):
        return [{"type": "text", "text": str(content)}]
    for block in content:
        if isinstance(block, dict):
            result.append(block)
        else:
            btype = getattr(block, "type", "unknown")
            if btype == "text":
                result.append({"type": "text", "text": getattr(block, "text", "")})
            elif btype == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": dict(getattr(block, "input", {})),
                })
            else:
                result.append({"type": btype, "raw": str(block)})
    return result
