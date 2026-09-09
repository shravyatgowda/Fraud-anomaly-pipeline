"""
features.py

Feature engineering for real-time fraud scoring.

Key design idea: computing "transaction velocity" (how many transactions
this user made recently) naively means a database query per incoming
transaction, which is slow under load. Instead, we keep a small,
thread-safe in-memory rolling window per user, updated as transactions
arrive, so feature lookups are O(1) dict/deque operations instead of
DB round-trips.

This is deliberately dependency-light (stdlib only) so it can be
unit-tested in isolation from the ML/serving stack.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Transaction:
    user_id: int
    amount: float
    timestamp: float  # unix seconds
    hour_of_day: int


class InMemoryVelocityStore:
    """
    Thread-safe rolling window of recent transaction timestamps per user,
    used to compute velocity-style features without hitting a database.
    """

    def __init__(self, window_seconds: int = 3600):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._history: dict[int, deque] = defaultdict(deque)

    def record_and_get_velocity(self, user_id: int, now: float | None = None) -> int:
        """
        Record a transaction for `user_id` at time `now` (defaults to
        current time) and return the number of transactions by that
        user within the trailing window, INCLUDING this one.
        """
        now = now if now is not None else time.time()
        with self._lock:
            dq = self._history[user_id]
            dq.append(now)
            cutoff = now - self.window_seconds
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)

    def clear(self):
        with self._lock:
            self._history.clear()


def compute_amount_deviation(amount: float, user_mean: float, user_std: float) -> float:
    """
    Z-score-style deviation of this transaction's amount from the
    user's historical average. Falls back to 0 std-dev safely.
    """
    if user_std <= 1e-9:
        return 0.0
    return (amount - user_mean) / user_std


def is_odd_hour(hour_of_day: int) -> bool:
    """Simple heuristic: transactions between midnight and 5 AM are unusual."""
    return hour_of_day in (0, 1, 2, 3, 4)


def build_feature_vector(
    amount: float,
    hour_of_day: int,
    velocity: int,
    user_mean: float,
    user_std: float,
) -> dict:
    """Assemble the final feature dict fed into the model."""
    return {
        "amount": amount,
        "amount_deviation": compute_amount_deviation(amount, user_mean, user_std),
        "hour_of_day": hour_of_day,
        "is_odd_hour": int(is_odd_hour(hour_of_day)),
        "txn_count_last_hour": velocity,
    }
