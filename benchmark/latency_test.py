"""
latency_test.py

Measures real end-to-end scoring latency (feature extraction + model
inference) over many iterations, to report a percentile latency figure
instead of a single anecdotal measurement.

Usage:
    python -m benchmark.latency_test
"""

import statistics
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import InMemoryVelocityStore, build_feature_vector


def run_benchmark(n_iterations: int = 5000):
    model_path = Path(__file__).resolve().parent.parent / "model.joblib"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_order = bundle["features"]

    store = InMemoryVelocityStore(window_seconds=3600)
    latencies_ms = []

    for i in range(n_iterations):
        start = time.perf_counter()

        velocity = store.record_and_get_velocity(user_id=i % 500)
        features = build_feature_vector(
            amount=100 + (i % 700),
            hour_of_day=i % 24,
            velocity=velocity,
            user_mean=500,
            user_std=120,
        )
        X = pd.DataFrame([[features[c] for c in feature_order]], columns=feature_order)
        _ = model.predict_proba(X)[0, 1]

        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]

    print(f"Iterations: {n_iterations}")
    print(f"Mean latency:   {statistics.mean(latencies_ms):.3f} ms")
    print(f"P50 latency:    {p50:.3f} ms")
    print(f"P95 latency:    {p95:.3f} ms")
    print(f"P99 latency:    {p99:.3f} ms")


if __name__ == "__main__":
    run_benchmark()
