#!/usr/bin/env python
"""CLI entrypoint for running an autolab agent session.

Usage:
    python scripts/run_session.py --session-config autolab/configs/session_demo.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an autolab agent session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-config",
        required=True,
        metavar="PATH",
        help="Path to the session YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from autolab.agent.session import SessionRunner, _load_config

    config_path = Path(args.session_config)
    if not config_path.exists():
        print(f"Error: session config not found: {config_path}", file=sys.stderr)
        return 1

    config = _load_config(config_path)
    runner = SessionRunner(config)
    terminal_status = runner.run()

    print(f"\nSession finished: {terminal_status}")
    return 0 if terminal_status in ("finalized", "human_interrupted") else 1


if __name__ == "__main__":
    sys.exit(main())
