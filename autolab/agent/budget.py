"""BudgetTracker — wall-clock compute budget for a training session."""

from __future__ import annotations

import time


class BudgetTracker:
    """Tracks elapsed wall-clock time against a fixed budget.

    Args:
        budget_seconds: Total allowed wall-clock seconds for the session.
    """

    def __init__(self, budget_seconds: float) -> None:
        self._start = time.monotonic()
        self.budget_seconds = budget_seconds

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since the tracker was created."""
        return time.monotonic() - self._start

    def remaining_seconds(self) -> float:
        """Seconds remaining. Never returns negative."""
        return max(0.0, self.budget_seconds - self.elapsed_seconds())

    def fraction_used(self) -> float:
        """Fraction of budget consumed, in [0, 1+]."""
        return self.elapsed_seconds() / self.budget_seconds

    def exhausted(self) -> bool:
        """True when elapsed >= budget."""
        return self.fraction_used() >= 1.0

    def near_exhausted(self, threshold: float = 0.9) -> bool:
        """True when fraction_used >= threshold (default 90%)."""
        return self.fraction_used() >= threshold
