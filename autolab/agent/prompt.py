"""System prompt rendering — substitutes [TOOL_REFERENCE] and [EXAMPLE_CONFIG]."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from autolab.control_plane.plane import ControlPlane

_TEMPLATE_PATH = Path(__file__).parent / "system_prompt.md"


def render_system_prompt(cp: "ControlPlane", initial_config: dict) -> str:
    """Load system_prompt.md and substitute the two placeholders.

    Args:
        cp: ControlPlane instance (provides tool_specs()).
        initial_config: The session's starting config dict (rendered as YAML).

    Returns:
        Fully rendered system prompt string.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    tool_ref = _format_tool_reference(cp.tool_specs())
    config_yaml = yaml.dump(initial_config, default_flow_style=False, allow_unicode=True)
    return (
        template
        .replace("[TOOL_REFERENCE]", tool_ref)
        .replace("[EXAMPLE_CONFIG]", f"```yaml\n{config_yaml}```")
    )


def _format_tool_reference(specs: list[dict]) -> str:
    """Format tool specs as a human-readable bullet list for the system prompt."""
    lines: list[str] = []
    write_tools = [
        "start_run", "fork", "kill", "modify", "set_active", "set_cadence", "eval",
    ]
    read_tools = [
        "get_run_details", "list_runs", "get_history", "get_sample",
        "compute_trend", "get_config", "get_eval",
    ]
    agent_tools = ["sleep", "finalize"]

    spec_by_name = {s["name"]: s for s in specs}

    def _render_group(names: list[str], header: str) -> None:
        lines.append(f"**{header}**")
        for name in names:
            spec = spec_by_name.get(name)
            if not spec:
                continue
            desc_first_line = spec["description"].split("\n")[0].strip().rstrip(".")
            props = spec["input_schema"].get("properties", {})
            required = spec["input_schema"].get("required", [])
            param_strs = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                if isinstance(ptype, list):
                    ptype = " | ".join(str(t) for t in ptype)
                req = "" if pname in required else "?"
                param_strs.append(f"{pname}{req}: {ptype}")
            params = ", ".join(param_strs) if param_strs else ""
            lines.append(f"- `{name}({params})` — {desc_first_line}")
        lines.append("")

    _render_group(write_tools, "Write tools (mutate state)")
    _render_group(read_tools, "Read tools (query only)")
    _render_group(agent_tools, "Agent-only tools")

    return "\n".join(lines)
