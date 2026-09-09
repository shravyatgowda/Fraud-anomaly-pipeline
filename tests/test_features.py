import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import (
    InMemoryVelocityStore,
    build_feature_vector,
    compute_amount_deviation,
    is_odd_hour,
)


def test_amount_deviation_zero_std():
    assert compute_amount_deviation(100, user_mean=50, user_std=0) == 0.0


def test_amount_deviation_normal_case():
    z = compute_amount_deviation(150, user_mean=100, user_std=25)
    assert z == 2.0


def test_is_odd_hour_true():
    assert is_odd_hour(2) is True


def test_is_odd_hour_false():
    assert is_odd_hour(14) is False


def test_velocity_store_counts_within_window():
    store = InMemoryVelocityStore(window_seconds=60)
    now = 1_000_000.0
    assert store.record_and_get_velocity(user_id=1, now=now) == 1
    assert store.record_and_get_velocity(user_id=1, now=now + 10) == 2
    assert store.record_and_get_velocity(user_id=1, now=now + 20) == 3


def test_velocity_store_expires_old_entries():
    store = InMemoryVelocityStore(window_seconds=60)
    now = 1_000_000.0
    store.record_and_get_velocity(user_id=1, now=now)
    store.record_and_get_velocity(user_id=1, now=now + 10)
    # This transaction is 90s after the first two -> they fall outside the 60s window
    velocity = store.record_and_get_velocity(user_id=1, now=now + 90)
    assert velocity == 1


def test_velocity_store_is_per_user():
    store = InMemoryVelocityStore(window_seconds=60)
    now = 1_000_000.0
    store.record_and_get_velocity(user_id=1, now=now)
    store.record_and_get_velocity(user_id=1, now=now)
    velocity_user_2 = store.record_and_get_velocity(user_id=2, now=now)
    assert velocity_user_2 == 1


def test_build_feature_vector_shape():
    features = build_feature_vector(
        amount=500, hour_of_day=2, velocity=3, user_mean=200, user_std=50
    )
    assert set(features.keys()) == {
        "amount", "amount_deviation", "hour_of_day", "is_odd_hour", "txn_count_last_hour"
    }
    assert features["is_odd_hour"] == 1


def test_feature_extraction_latency_is_fast():
    """
    Sanity-check that a single feature-extraction call (velocity lookup +
    vector assembly) comfortably clears a 15ms budget on this machine —
    this is a smoke test, not a formal benchmark (see benchmark/latency_test.py).
    """
    store = InMemoryVelocityStore(window_seconds=3600)
    start = time.perf_counter()
    velocity = store.record_and_get_velocity(user_id=42)
    features = build_feature_vector(
        amount=1200, hour_of_day=10, velocity=velocity, user_mean=800, user_std=150
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 15
    assert features["txn_count_last_hour"] == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
