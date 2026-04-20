"""HumanMessageQueue — in-memory queue backed by an inbox.txt file tail."""

from __future__ import annotations

import queue
from pathlib import Path


class HumanMessageQueue:
    """Delivers human messages to the agent loop via a file-tail reader.

    The human drops lines into ``inbox_path`` (plain text, one message per line).
    ``poll()`` reads any new lines since the last call and puts them onto an
    in-memory queue.  The loop calls ``poll()`` at the top of each iteration and
    inside the sleep scheduler's polling loop.

    Args:
        inbox_path: Path to the inbox file. Created automatically on first write
            by the human; no error if it does not yet exist.
    """

    def __init__(self, inbox_path: Path) -> None:
        self._inbox_path = Path(inbox_path)
        self._queue: queue.Queue[str] = queue.Queue()
        self._file_pos: int = 0

    def poll(self) -> None:
        """Read any new lines from inbox.txt since the last poll.

        Safe to call even if the file does not exist.
        """
        if not self._inbox_path.exists():
            return
        try:
            with self._inbox_path.open("r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                for line in f:
                    stripped = line.rstrip("\n").rstrip("\r")
                    if stripped:
                        self._queue.put(stripped)
                self._file_pos = f.tell()
        except OSError:
            pass

    def empty(self) -> bool:
        """True if no messages are waiting."""
        return self._queue.empty()

    def drain(self) -> list[str]:
        """Return and remove all queued messages."""
        messages: list[str] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages
