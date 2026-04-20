#!/usr/bin/env python3
"""Build a ReplayFile JSON from backend artifacts for the autolab-replay-dashboard.

Usage:
    python scripts/build_replay.py \\
        --session-id phase-c-014 \\
        [--db store/autolab.db] \\
        [--logs-dir logs/] \\
        [--sessions-dir sessions/] \\
        [--output autolab-replay-dashboard/public/replays/]

Output:
    {output}/{session_id}.json   — ReplayFile consumed by the frontend
    {output}/index.json          — Updated list of all .json files in output dir
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Tool categories (must match frontend expectation)
# ---------------------------------------------------------------------------

_CATEGORIES: dict[str, str] = {
    "start_run": "write",
    "fork": "write",
    "kill": "write",
    "modify": "write",
    "eval": "write",
    "set_active": "write",
    "set_cadence": "control",
    "sleep": "control",
    "get_run_details": "read",
    "list_runs": "read",
    "get_history": "read",
    "get_sample": "read",
    "compute_trend": "read",
    "get_config": "read",
    "get_eval": "read",
    "finalize": "finalize",
}

_STATUS_MAP = {"done": "completed"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso_ts(ts_str: str) -> float:
    """Convert ISO 8601 UTC string to Unix seconds."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def _summarize_result(tool_name: str, content: dict) -> str:
    err = content.get("error")
    if err:
        if isinstance(err, dict):
            return f"error: {err.get('code', '?')}: {err.get('message', '')[:80]}"
        return f"error: {str(err)[:80]}"

    r = content.get("result") or {}

    if tool_name == "start_run":
        return f"run started: {r.get('run_id', '?')}"
    if tool_name == "modify":
        return f"modify applied at step {r.get('step', '?')}: {r.get('applied', [])}"
    if tool_name == "fork":
        return f"forked to {r.get('run_id', '?')} at step {r.get('fork_step', '?')}"
    if tool_name == "kill":
        return f"killed: {r.get('run_id', '?')}"
    if tool_name == "set_cadence":
        return (
            f"cadence set: obs={r.get('observation_cadence', '?')}, "
            f"sample={r.get('sample_cadence', '?')}"
        )
    if tool_name == "compute_trend":
        if r.get("slope") is not None:
            return f"slope={r['slope']:.4f}, mean={r['mean']:.3f}, std={r['std']:.3f}"
        return r.get("note", "no data")
    if tool_name == "sleep":
        return f"slept {r.get('actual_seconds', 0):.0f}s, woken by {r.get('woken_by', '?')}"
    if tool_name == "eval":
        return f"eval started: {r.get('eval_id', '?')}"
    if tool_name == "get_eval":
        if isinstance(r, dict) and r.get("accuracy") is not None:
            return f"accuracy={r['accuracy']:.3f}"
        return f"status: {r.get('status', '?')}"
    if tool_name == "finalize":
        return "session finalized"
    if tool_name == "set_active":
        return f"active: {r.get('active_run_id', 'none')}"

    return json.dumps(r)[:100] if r else "ok"


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_replay(
    session_id: str,
    db_path: str,
    logs_dir: str,
    sessions_dir: str,
    output_dir: str,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── Session ──────────────────────────────────────────────────────────────
    session_row = conn.execute(
        "SELECT * FROM sessions WHERE session_id=?", [session_id]
    ).fetchone()
    if session_row is None:
        print(f"Session '{session_id}' not found in {db_path}", file=sys.stderr)
        sys.exit(1)

    # ── System prompt ────────────────────────────────────────────────────────
    sp_path = Path(sessions_dir) / session_id / "system_prompt.txt"
    system_prompt = sp_path.read_text(encoding="utf-8") if sp_path.exists() else ""
    if not system_prompt:
        print("Warning: system_prompt.txt not found — system_prompt will be empty.")

    # ── Initial config ───────────────────────────────────────────────────────
    initial_config_yaml = ""
    if session_row["initial_config_id"]:
        cfg_row = conn.execute(
            "SELECT config_json FROM configs WHERE config_id=?",
            [session_row["initial_config_id"]],
        ).fetchone()
        if cfg_row:
            initial_config = json.loads(cfg_row["config_json"])
            initial_config_yaml = yaml.dump(initial_config, default_flow_style=False, allow_unicode=True)

    # ── First pass: collect tool_results + discover run_ids via conversation ─
    conv_path = Path(sessions_dir) / session_id / "conversation.jsonl"
    if not conv_path.exists():
        print(f"conversation.jsonl not found at {conv_path}", file=sys.stderr)
        sys.exit(1)

    tool_results: dict[str, dict] = {}   # tool_use_id → {name, content}
    discovered_run_ids: set[str] = set()
    raw_messages: list[dict] = []

    for line in conv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        raw_messages.append(msg)
        if msg["role"] == "tool_result":
            tool_results[msg["tool_use_id"]] = {
                "name": msg["tool_name"],
                "content": msg["content"],
            }
            result = (msg["content"].get("result") or {})
            if msg["tool_name"] in ("start_run", "fork") and result.get("run_id"):
                discovered_run_ids.add(result["run_id"])

    # Belt-and-suspenders: also query DB for runs in this session's time window
    ended_at = session_row["ended_at"] or 9_999_999_999.0
    for row in conn.execute(
        "SELECT run_id FROM runs WHERE start_time >= ? AND start_time <= ?",
        [session_row["started_at"], ended_at],
    ).fetchall():
        discovered_run_ids.add(row["run_id"])

    if not discovered_run_ids:
        print("Warning: no runs discovered for this session.")

    # ── Observations from logs/ ───────────────────────────────────────────────
    all_observations: list[dict] = []
    for run_id in discovered_run_ids:
        log_path = Path(logs_dir) / f"{run_id}.jsonl"
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obs = json.loads(line)
            ts = _parse_iso_ts(obs["timestamp"])
            all_observations.append({
                "type": "observation",
                "ts": ts,
                "run_id": obs["run_id"],
                "step": obs["step"],
                "metrics": obs["metrics"],
                "trends": obs["trends"],
                "anomalies": obs["anomalies"],
                "is_anomaly": obs["is_anomaly"],
            })

    # ── Modifications from DB (for diff construction) ─────────────────────────
    mods_by_run: dict[str, list[dict]] = defaultdict(list)
    for run_id in discovered_run_ids:
        for row in conn.execute(
            "SELECT * FROM modifications WHERE run_id=? ORDER BY step ASC", [run_id]
        ).fetchall():
            old_cfg_row = conn.execute(
                "SELECT config_json FROM configs WHERE config_id=?", [row["old_config_id"]]
            ).fetchone()
            if old_cfg_row:
                old_cfg = json.loads(old_cfg_row["config_json"])
                mods_by_run[run_id].append({
                    "step": row["step"],
                    "changes": json.loads(row["changes"]),
                    "old_config": old_cfg,
                })

    # ── Second pass: build conversation events ────────────────────────────────
    is_first_user = True
    conv_events: list[dict] = []

    for msg in raw_messages:
        ts = float(msg["ts"])
        role = msg["role"]

        if role == "user":
            text = msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
            etype = "charter" if is_first_user else "human_message"
            is_first_user = False
            conv_events.append({"type": etype, "ts": ts, "text": text})

        elif role == "assistant":
            content = msg["content"]
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        conv_events.append({"type": "agent_text", "ts": ts, "text": text})

                elif btype == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input") or {}
                    tr = tool_results.get(block.get("id", ""), {})
                    result = (tr.get("content") or {}).get("result") or {}

                    # step: read from tool_result where possible
                    step = None
                    if name == "modify":
                        step = result.get("step")
                    elif name == "fork":
                        step = result.get("fork_step")
                    elif name != "start_run":
                        run_id_ctx = inp.get("run_id") or inp.get("parent_run_id")
                        if run_id_ctx:
                            prior = [o for o in all_observations
                                     if o["run_id"] == run_id_ctx and o["ts"] <= ts]
                            if prior:
                                step = max(prior, key=lambda o: o["ts"])["step"]

                    # diff
                    diff = None
                    rid = inp.get("run_id") or inp.get("parent_run_id")
                    if name == "modify" and rid:
                        mod_step = result.get("step")
                        matching = [m for m in mods_by_run.get(rid, [])
                                    if m["step"] == mod_step]
                        if matching:
                            m = matching[0]
                            diff = [
                                {
                                    "param": k,
                                    "old": m["old_config"].get("grpo", {}).get(k),
                                    "new": v,
                                }
                                for k, v in m["changes"].items()
                            ]
                    elif name == "fork":
                        overrides = inp.get("overrides") or {}
                        if rid and overrides:
                            p_run_row = conn.execute(
                                "SELECT config_id FROM runs WHERE run_id=?", [rid]
                            ).fetchone()
                            if p_run_row:
                                parent_cfg_row = conn.execute(
                                    "SELECT config_json FROM configs WHERE config_id=?",
                                    [p_run_row["config_id"]],
                                ).fetchone()
                                if parent_cfg_row:
                                    parent_cfg = json.loads(parent_cfg_row["config_json"])
                                    diff = []
                                    for section, params in overrides.items():
                                        if isinstance(params, dict):
                                            for k, v in params.items():
                                                diff.append({
                                                    "param": k,
                                                    "old": parent_cfg.get(section, {}).get(k),
                                                    "new": v,
                                                })
                                        else:
                                            diff.append({
                                                "param": section,
                                                "old": parent_cfg.get(section),
                                                "new": params,
                                            })

                    # run_id field: use child (created by call) for start_run and fork
                    if name == "start_run":
                        event_run_id = result.get("run_id")
                    elif name == "fork":
                        event_run_id = result.get("run_id")  # child run
                    else:
                        event_run_id = inp.get("run_id") or inp.get("parent_run_id")

                    conv_events.append({
                        "type": "tool_call",
                        "ts": ts,
                        "tool_use_id": block.get("id", ""),
                        "name": name,
                        "input": inp,
                        "reason": inp.get("reason", ""),
                        "run_id": event_run_id,
                        "step": step,
                        "diff": diff,
                        "category": _CATEGORIES.get(name, "read"),
                    })

        elif role == "tool_result":
            c = msg["content"] or {}
            err = c.get("error")
            ok = err is None
            if isinstance(err, dict):
                error_code = err.get("code")
            elif err:
                error_code = str(err)[:50]
            else:
                error_code = None

            conv_events.append({
                "type": "tool_result",
                "ts": ts,
                "tool_use_id": msg.get("tool_use_id", ""),
                "ok": ok,
                "error_code": error_code,
                "summary": _summarize_result(msg.get("tool_name", ""), c),
            })

    # ── Runs ─────────────────────────────────────────────────────────────────
    runs: list[dict] = []
    for run_id in sorted(discovered_run_ids):  # stable order
        run_row = conn.execute(
            "SELECT * FROM runs WHERE run_id=?", [run_id]
        ).fetchone()
        if run_row is None:
            continue
        cfg_row = conn.execute(
            "SELECT config_json FROM configs WHERE config_id=?", [run_row["config_id"]]
        ).fetchone()
        run_cfg = json.loads(cfg_row["config_json"]) if cfg_row else {}
        obs_for_run = [o for o in all_observations if o["run_id"] == run_id]
        status = _STATUS_MAP.get(run_row["status"], run_row["status"])
        runs.append({
            "run_id": run_id,
            "parent_run_id": run_row["parent_run_id"],
            "fork_step": run_row["fork_step"],
            "initial_config": run_cfg,
            "status": status,
            "started_step": min((o["step"] for o in obs_for_run), default=0),
            "ended_step": max((o["step"] for o in obs_for_run), default=None),
            "creation_reason": run_row["creation_reason"] or "",
        })

    # ── Merge and sort all events by timestamp ────────────────────────────────
    all_events = sorted(conv_events + all_observations, key=lambda e: e["ts"])

    # ── Assemble replay ────────────────────────────────────────────────────────
    replay = {
        "session": {
            "id": session_row["session_id"],
            "goal": session_row["goal"],
            "system_prompt": system_prompt,
            "initial_config_yaml": initial_config_yaml,
            "budget_seconds": session_row["budget_seconds"],
            "started_at": session_row["started_at"],
            "ended_at": session_row["ended_at"],
        },
        "runs": runs,
        "events": all_events,
    }

    # ── Write output ──────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{session_id}.json"
    out_path.write_text(json.dumps(replay, indent=2, default=str), encoding="utf-8")
    print(f"Written: {out_path}  ({out_path.stat().st_size // 1024}KB)")

    # Update index.json
    index_path = out_dir / "index.json"
    existing: list[str] = json.loads(index_path.read_text()) if index_path.exists() else []
    fname = f"{session_id}.json"
    if fname not in existing:
        existing.append(fname)
    index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Updated: {index_path}")

    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay JSON from autolab session artifacts.")
    parser.add_argument("--session-id", required=True, help="session_id to build replay for")
    parser.add_argument("--db", default="store/autolab.db", help="Path to SQLite DB")
    parser.add_argument("--logs-dir", default="logs/", help="Directory containing {run_id}.jsonl files")
    parser.add_argument("--sessions-dir", default="sessions/", help="Directory containing session subdirs")
    parser.add_argument(
        "--output",
        default="autolab-replay-dashboard/public/replays/",
        help="Output directory for replay JSON and index.json",
    )
    args = parser.parse_args()

    build_replay(
        session_id=args.session_id,
        db_path=args.db,
        logs_dir=args.logs_dir,
        sessions_dir=args.sessions_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
