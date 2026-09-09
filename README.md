# Real-Time Fraud & Anomaly Detection Pipeline

A personal project exploring how to make fraud scoring fast enough for
a live payment path — not just accurate in a notebook, but cheap enough
per-request to sit in the critical path of a transaction.

The core idea: naive fraud-scoring pipelines often compute features
(like "how many transactions has this user made recently?") with a
database query per incoming transaction. Under load, that DB round-trip
becomes the bottleneck, not the model. This project replaces that
lookup with a thread-safe **in-memory rolling window** per user, so
feature extraction is O(1) instead of a network call.

## Results (measured on this machine, synthetic data — see caveats below)

Generated with `python data/generate_synthetic_data.py` (20,000 transactions,
2% synthetic fraud rate) and `python -m src.train`:

```
              precision    recall  f1-score   support
           0      0.999     0.999     0.999      4900
           1      0.950     0.960     0.955       100
ROC-AUC: 1.000
```

Latency, measured with `python -m benchmark.latency_test` (5,000 scored
transactions, feature extraction + model inference, single process):

```
Mean latency:   11.4 ms
P50 latency:    11.1 ms
P95 latency:    13.2 ms
P99 latency:    14.7 ms
```

**Caveats, read before quoting these numbers anywhere:** this is a
synthetic dataset generated to have a clearly learnable fraud pattern
(see `data/generate_synthetic_data.py`), so precision/recall here are
optimistic compared to real transaction data, which is messier and has
harder-to-separate fraud patterns. Latency numbers are single-process,
single-machine, no real network/database in the loop, and no concurrent
load — a production benchmark would use a load-testing tool (e.g. `locust`
or `wrk`) against the running FastAPI server under concurrency.

## Architecture

```
fraud-anomaly-pipeline/
├── data/
│   └── generate_synthetic_data.py   # synthetic transaction generator
├── src/
│   ├── features.py                  # in-memory velocity store + feature engineering
│   └── train.py                     # trains & evaluates RandomForestClassifier
├── api/
│   ├── main.py                      # FastAPI async scoring endpoint
│   └── schemas.py                   # request/response models
├── benchmark/
│   └── latency_test.py              # P50/P95/P99 latency measurement
├── tests/
│   └── test_features.py             # unit tests for feature engineering
└── requirements.txt
```

### Feature engineering

- **Velocity** (`txn_count_last_hour`): tracked via `InMemoryVelocityStore`,
  a thread-safe per-user deque of recent timestamps, pruned lazily on each
  access — no external cache/DB needed for this signal.
- **Amount deviation**: a running per-user mean/std computed online via
  **Welford's algorithm**, so it updates in O(1) per transaction without
  storing full transaction history in memory.
- **Odd-hour flag**: simple heuristic (midnight–5AM), cheap and interpretable.

### Model

A `RandomForestClassifier` (scikit-learn), chosen for a reasonable balance
of accuracy and interpretability (feature importances) over something like
a neural net, given the small feature set and tabular data.

### Serving

FastAPI, model loaded once at startup (not per-request). The scoring
endpoint is `async def` so uvicorn can interleave many in-flight requests;
inference itself is fast enough (sub-millisecond for a shallow forest on
5 features) that it doesn't block the event loop meaningfully at moderate
throughput.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate synthetic data
python data/generate_synthetic_data.py

# 2. Train the model
python -m src.train

# 3. Run the API
uvicorn api.main:app --reload

# 4. Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 1, "user_id": 42, "amount": 5000, "hour_of_day": 3}'

# 5. Run tests
pytest tests/ -v

# 6. Run the latency benchmark
python -m benchmark.latency_test
```

## Design trade-offs

- **In-memory state vs. a real feature store**: this trades durability
  for speed — if the process restarts, per-user history resets. A
  production system would back this with something like Redis for
  durability while keeping the same low-latency access pattern.
- **Single-process benchmark**: real production latency depends heavily
  on concurrent load, network overhead, and whether the model runs on
  the same host as the API — this benchmark isolates just the
  feature-extraction + inference cost.
- **Random Forest over deep learning**: chosen for fast inference and
  interpretable feature importances on a small tabular feature set —
  worth revisiting if the feature set grows much larger or includes
  sequence data.

## Possible extensions

- Swap the in-memory store for Redis so state survives restarts and can
  be shared across multiple API instances.
- Add a proper load test (`locust`/`wrk`) against the running server to
  get realistic concurrent-throughput latency numbers.
- Add SHAP-based explanations per flagged transaction for analyst review.

## Tech stack

Python · scikit-learn · pandas · FastAPI · pytest
