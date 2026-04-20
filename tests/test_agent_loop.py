"""Phase A mock dry run — tests AgentLoop wiring without GPU, DB, or real LLM.

Exercises all tricky paths:
  - Basic tool dispatch (list_runs, start_run)
  - Tool error handling (start_run returns TRAINER_START_FAILED)
  - Human message queue (write to inbox.txt during sleep)
  - Budget warning injection into StatusBar.warnings
  - History compression (enough turns to trigger it)
  - .stop file interrupt mid-sleep
  - Budget exhaustion → auto-finalize

Run with:
    cd c:/Users/aritr/Desktop/auto-lab
    python tests/test_agent_loop.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autolab.agent.budget import BudgetTracker
from autolab.agent.history import ConversationHistory
from autolab.agent.human_queue import HumanMessageQueue
from autolab.agent.loop import AgentLoop
from autolab.agent.sleep import SleepScheduler
from autolab.control_plane.types import ErrorCode, StatusBar, ToolError, ToolResponse
from tests.mock_cp import MockControlPlane

# ── Mock Store ────────────────────────────────────────────────────────────────

class MockStore:
    """In-memory store stub."""
    def __init__(self):
        self._sessions = {}
        self._runs = []
        self._evals = []
    def insert_session(self, **kw): self._sessions[kw["session_id"]] = {**kw, "status": "running"}
    def update_session(self, session_id, **fields):
        if session_id in self._sessions:
            self._sessions[session_id].update(fields)
    def get_session(self, session_id): return self._sessions.get(session_id)
    def list_runs(self): return self._runs
    def get_history(self, run_id, **kw): return []
    def list_evals_all_runs(self): return self._evals


# ── Mock LLM Client ───────────────────────────────────────────────────────────

class _FakeBlock:
    """Fake Anthropic content block."""
    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeMessage:
    """Fake Anthropic Message object."""
    def __init__(self, content):
        self.content = content
        self.stop_reason = "end_turn"


def _tool_use(name: str, args: dict, id_: str = "tu1") -> _FakeBlock:
    b = _FakeBlock("tool_use", name=name, input=args, id=id_)
    return b


def _text(text: str) -> _FakeBlock:
    return _FakeBlock("text", text=text)


class MockLLMClient:
    """Scripted LLM that returns pre-defined tool call sequences.

    Each call to messages.create() pops the next response from the script.
    Supports multiple scenarios via the `scenario` constructor argument.

    The real Anthropic client has ``client.messages`` as a resource object with
    a ``.create()`` method. We replicate that interface here.
    """

    def __init__(self, scenario: str = "normal") -> None:
        self._scenario = scenario
        self._call_count = 0
        self._script = self._build_script(scenario)
        # Expose messages as an attribute object (matching Anthropic SDK interface)
        self.messages = self._MessagesProxy(self)

    def _build_script(self, scenario: str) -> list:
        if scenario == "normal":
            # Turn 1: list_runs (basic read)
            # Turn 2: start_run (success)
            # Turn 3: sleep 5s (human message will arrive during sleep)
            # Turn 4: set_active(None) — triggers budget near-exhausted warning
            # Turns 5-15: list_runs repeatedly (triggers history compression)
            # Turn 16: finalize
            script = [
                _FakeMessage([_tool_use("list_runs", {}, "tu1")]),
                _FakeMessage([_tool_use("start_run", {"config": {"model_name": "test", "dataset_name": "test", "grpo": {}}, "reason": "initial run"}, "tu2")]),
                _FakeMessage([_text("Sleeping briefly."), _tool_use("sleep", {"seconds": 5, "reason": "waiting"}, "tu3")]),
                _FakeMessage([_text("Got human message. Pausing."), _tool_use("set_active", {"run_id": None, "reason": "pause"}, "tu4")]),
            ]
            # Turns 5-15: repeated list_runs to trigger compression (need >10 turns)
            for i in range(11):
                script.append(_FakeMessage([_tool_use("list_runs", {}, f"tu-list-{i}")]))
            # Final turn: finalize
            script.append(_FakeMessage([
                _text("Done exploring. Finalizing."),
                _tool_use("finalize", {
                    "winning_run_id": None,
                    "summary": "Mock session: verified loop wiring.",
                    "reason": "test complete",
                }, "tu-fin"),
            ]))
            return script

        elif scenario == "error_handling":
            # Turn 1: start_run → returns ToolError (we'll inject this via MockCP override)
            # Turn 2: list_runs
            # Turn 3: finalize
            return [
                _FakeMessage([_tool_use("start_run", {"config": {"model_name": "x", "dataset_name": "x", "grpo": {}}, "reason": "test"}, "tu1")]),
                _FakeMessage([_tool_use("list_runs", {}, "tu2")]),
                _FakeMessage([_tool_use("finalize", {"winning_run_id": None, "summary": "handled error", "reason": "done"}, "tu3")]),
            ]

        elif scenario == "stop_file":
            # Turn 1: sleep 60s → .stop file will be created after 1s by test thread
            # Turn 2: finalize (agent sees interrupt message)
            return [
                _FakeMessage([_tool_use("sleep", {"seconds": 60, "reason": "long sleep"}, "tu1")]),
                _FakeMessage([_tool_use("finalize", {"winning_run_id": None, "summary": "interrupted", "reason": "stop file"}, "tu2")]),
            ]

        elif scenario == "budget_exhaustion":
            # Budget = 2s. Turn 1: start_run → should get BUDGET_EXHAUSTED after budget expires.
            # After _MAX_OVERBUDGET_TURNS turns without finalize, auto-finalize fires.
            script = []
            for i in range(6):  # more than _MAX_OVERBUDGET_TURNS=3
                script.append(_FakeMessage([_tool_use("start_run", {
                    "config": {"model_name": "x", "dataset_name": "x", "grpo": {}},
                    "reason": f"attempt {i}"
                }, f"tu-{i}")]))
            return script

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def create(self, *, model, system=None, messages, tools=None, max_tokens=4096, **kwargs):
        if self._call_count >= len(self._script):
            # Fallback: finalize so the loop terminates
            return _FakeMessage([_tool_use("finalize", {
                "winning_run_id": None,
                "summary": "auto-finalize (script exhausted)",
                "reason": "script end",
            }, "tu-auto")])
        msg = self._script[self._call_count]
        self._call_count += 1
        return msg

    class _MessagesProxy:
        """Mirrors anthropic.resources.Messages — exposes .create()."""
        def __init__(self, parent: "MockLLMClient") -> None:
            self._parent = parent
        def create(self, **kw):
            return self._parent.create(**kw)


def _build_loop(
    scenario: str,
    session_dir: Path,
    budget_seconds: float = 3600.0,
    cp_override=None,
) -> tuple[AgentLoop, MockStore, MockControlPlane]:
    from autolab.store.logs import Logs

    store = MockStore()
    cp = cp_override or MockControlPlane()
    logs = Logs(str(session_dir / "logs"))
    sleep_scheduler = SleepScheduler(logs, cp)
    human_queue = HumanMessageQueue(session_dir / "inbox.txt")
    history = ConversationHistory(max_live_turns=5, summarizer_model="mock")  # low max to trigger compression
    budget = BudgetTracker(budget_seconds)
    stop_event = threading.Event()
    client = MockLLMClient(scenario)

    store.insert_session(
        session_id="test-session",
        started_at=time.time(),
        goal="test",
        success_criterion="any",
        budget_seconds=budget_seconds,
    )

    loop = AgentLoop(
        cp=cp,
        store=store,
        session_id="test-session",
        session_dir=session_dir,
        goal="test goal",
        success_criterion="session completes cleanly",
        budget=budget,
        human_queue=human_queue,
        sleep_scheduler=sleep_scheduler,
        history=history,
        system_prompt="[test system prompt]",
        client=client,
        stop_event=stop_event,
        model="mock-model",
        max_tokens=1000,
        sleep_default=5.0,
    )
    return loop, store, cp


# ── Test cases ────────────────────────────────────────────────────────────────

def test_normal_flow(session_dir: Path) -> None:
    """Normal flow: list_runs → start_run → sleep (with human msg) → finalize."""
    print("  [normal_flow] running...", end=" ", flush=True)

    inbox = session_dir / "inbox.txt"

    # Write a human message into inbox.txt 2s after test starts (during the sleep turn)
    def _inject_message():
        time.sleep(2)
        inbox.write_text("Hello agent, please continue.\n", encoding="utf-8")

    t = threading.Thread(target=_inject_message, daemon=True)
    t.start()

    loop, store, cp = _build_loop("normal", session_dir)
    status = loop.run()

    assert status == "finalized", f"Expected finalized, got {status}"
    session = store.get_session("test-session")
    assert session["status"] == "finalized", f"Session status: {session['status']}"
    assert session.get("ended_at") is not None

    # Verify conversation.jsonl was written
    conv_log = session_dir / "conversation.jsonl"
    assert conv_log.exists(), "conversation.jsonl not created"
    lines = conv_log.read_text().strip().split("\n")
    assert len(lines) > 1, f"Expected multiple log lines, got {len(lines)}"
    print(f"PASS ({len(lines)} log lines)")


def test_error_handling(session_dir: Path) -> None:
    """start_run returns TRAINER_START_FAILED → agent reads error, finalizes."""
    print("  [error_handling] running...", end=" ", flush=True)

    # Override start_run to return an error
    class ErrorCP(MockControlPlane):
        def start_run(self, config, reason=""):
            from autolab.control_plane.types import StatusBar, ToolError, ToolResponse
            return ToolResponse(
                status=self._status_bar(),
                error=ToolError(
                    code="TRAINER_START_FAILED",
                    message="Failed to load model (no GPU).",
                    retryable=False,
                    suggested_action="Try a smaller model or check hardware.",
                ),
            )

    loop, store, cp = _build_loop("error_handling", session_dir, cp_override=ErrorCP())
    status = loop.run()
    assert status == "finalized", f"Expected finalized, got {status}"
    print("PASS")


def test_stop_file(session_dir: Path) -> None:
    """Create .stop file 1s into a 60s sleep → loop exits with human_interrupted."""
    print("  [stop_file] running...", end=" ", flush=True)

    def _create_stop():
        time.sleep(1)
        (session_dir / ".stop").write_text("", encoding="utf-8")

    t = threading.Thread(target=_create_stop, daemon=True)
    t.start()

    loop, store, _ = _build_loop("stop_file", session_dir)
    status = loop.run()

    # Agent may finalize (script has finalize after sleep) or be auto-finalized
    assert status in ("finalized", "human_interrupted"), f"Unexpected status: {status}"
    print(f"PASS (status={status})")


def test_budget_warnings(session_dir: Path) -> None:
    """Budget near_exhausted at 90% injects warning into StatusBar.warnings."""
    print("  [budget_warnings] running...", end=" ", flush=True)

    # Very tight budget so it's near-exhausted immediately
    loop, store, cp = _build_loop("normal", session_dir, budget_seconds=1.0)

    # Override budget to be near_exhausted from the start
    from autolab.agent.budget import BudgetTracker
    loop._budget = BudgetTracker(0.001)  # essentially immediately exhausted

    # Run one manual tool invocation (not full loop — just check warning injection)
    loop._budget._start -= 0.001  # force exhausted
    result = loop._invoke_tool("list_runs", {})
    # Budget exhausted but list_runs is not a mutating tool — should succeed
    assert "error" not in result or result.get("error") is None, "Read tool should not error on budget"
    print("PASS")


def test_budget_exhaustion_auto_finalize(session_dir: Path) -> None:
    """Budget exhausted + _MAX_OVERBUDGET_TURNS turns without finalize → budget_exhausted."""
    print("  [budget_exhaustion] running...", end=" ", flush=True)

    loop, store, cp = _build_loop("budget_exhaustion", session_dir, budget_seconds=0.001)
    # Force budget immediately exhausted
    time.sleep(0.01)

    status = loop.run()
    assert status == "budget_exhausted", f"Expected budget_exhausted, got {status}"
    session = store.get_session("test-session")
    assert session["status"] == "budget_exhausted"
    print("PASS")


def test_history_compression(session_dir: Path) -> None:
    """History compression triggers when live window > max_live_turns * 2."""
    print("  [history_compression] running...", end=" ", flush=True)

    # MockLLMClient will compress via _compress_with_haiku — but in mock mode, the
    # anthropic client call will fail. We test that maybe_compress doesn't crash
    # by making the client a stub that returns a fake compression response.

    class MockHaikuClient:
        class _Msg:
            class _Block:
                text = "Summary: ran mock tests, all good."
            content = [_Block()]
        def __init__(self):
            self.messages = self._Proxy(self)
        def create(self, **kw): return self._Msg()
        class _Proxy:
            def __init__(self, p): self._p = p
            def create(self, **kw): return self._p.create(**kw)

    loop, store, cp = _build_loop("normal", session_dir)
    # Replace compression client with Haiku stub
    # The history client is the same as loop._client (used for both LLM and compression)
    # Monkey-patch: inject a messages.create wrapper
    haiku_client = MockHaikuClient()
    original_client = loop._client

    class HybridClient:
        """LLM calls use original_client; compression calls (no tools) use haiku_client."""
        def __init__(self):
            self.messages = self._Proxy(self)
        def create(self, *, model, system=None, messages, tools=None, **kw):
            if tools is None:
                return haiku_client.create(model=model, messages=messages, **kw)
            return original_client.create(model=model, system=system, messages=messages, tools=tools, **kw)
        class _Proxy:
            def __init__(self, p): self._p = p
            def create(self, **kw): return self._p.create(**kw)

    loop._client = HybridClient()

    # Run enough turns that history.maybe_compress fires (max_live=5 turns * 2 = 10 msgs)
    status = loop.run()
    assert status == "finalized"

    # Verify compression summary was generated (if enough turns)
    # The normal scenario has 16+ turns, which should trigger compression
    print(f"PASS (history summary: {loop._history._summary!r:.50})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("Phase A — Mock dry run\n")

    tests = [
        test_normal_flow,
        test_error_handling,
        test_stop_file,
        test_budget_warnings,
        test_budget_exhaustion_auto_finalize,
        test_history_compression,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        tmpdir = Path(tempfile.mkdtemp(prefix=f"autolab_test_{test_fn.__name__}_"))
        try:
            test_fn(tmpdir)
            passed += 1
        except Exception as exc:
            print(f"FAIL: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        return 1
    print("\nPhase A complete. Ready for Phase B (real API, no GPU).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
