"""
main.py

FastAPI service that scores incoming transactions for fraud in
(near) real time.

Design choices, on purpose:
  - Model is loaded ONCE at startup, not per request.
  - Per-user velocity/mean/std stats are held in-memory (see
    src/features.py InMemoryVelocityStore + a simple running-stats
    dict) instead of querying a database per request, since that
    round-trip would dominate scoring latency under load.
  - The endpoint is declared `async def` so the ASGI server (uvicorn)
    can interleave many in-flight requests; the actual model.predict
    call is CPU-bound and fast (a few hundred microseconds for a
    shallow forest on 5 features), so it doesn't block the event loop
    long enough to matter at moderate throughput. For much heavier
    models this would move to a thread/process pool.

Run with:
    uvicorn api.main:app --reload
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import InMemoryVelocityStore, build_feature_vector
from api.schemas import ScoreOut, TransactionIn

app = FastAPI(
    title="Real-Time Fraud Scoring API",
    description="Scores transactions for fraud using in-memory feature lookups.",
    version="0.1.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
FRAUD_THRESHOLD = 0.5

_model_bundle = None
_velocity_store = InMemoryVelocityStore(window_seconds=3600)
# Running per-user amount stats: user_id -> (count, mean, M2) via Welford's algorithm
_user_stats: dict[int, list] = {}


@app.on_event("startup")
def load_model():
    global _model_bundle
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    _model_bundle = joblib.load(MODEL_PATH)


def _update_and_get_user_stats(user_id: int, amount: float) -> tuple[float, float]:
    """Welford's online algorithm: O(1) mean/variance update, no history array needed."""
    stats = _user_stats.setdefault(user_id, [0, 0.0, 0.0])  # [count, mean, M2]
    stats[0] += 1
    count, mean, m2 = stats
    delta = amount - mean
    mean += delta / count
    delta2 = amount - mean
    m2 += delta * delta2
    _user_stats[user_id] = [count, mean, m2]
    variance = m2 / count if count > 1 else 0.0
    return mean, variance**0.5


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/score", response_model=ScoreOut)
async def score_transaction(txn: TransactionIn):
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()

    velocity = _velocity_store.record_and_get_velocity(txn.user_id)
    user_mean, user_std = _update_and_get_user_stats(txn.user_id, txn.amount)

    features = build_feature_vector(
        amount=txn.amount,
        hour_of_day=txn.hour_of_day,
        velocity=velocity,
        user_mean=user_mean,
        user_std=user_std,
    )

    model = _model_bundle["model"]
    feature_order = _model_bundle["features"]
    X = pd.DataFrame([[features[c] for c in feature_order]], columns=feature_order)

    fraud_proba = float(model.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreOut(
        transaction_id=txn.transaction_id,
        fraud_probability=round(fraud_proba, 4),
        is_flagged=fraud_proba >= FRAUD_THRESHOLD,
        features_used=features,
        latency_ms=round(latency_ms, 3),
    )
