"""
generate_synthetic_data.py

Generates a synthetic financial transaction dataset with a small
fraction of labeled fraud cases, for training/testing the anomaly
detection pipeline without needing real (sensitive) transaction data.

Fraud transactions are simulated to have statistically distinct
patterns (unusual amount, odd hour, high velocity) so the model has
learnable signal — this is a teaching/demo dataset, not a claim about
real-world fraud distributions.
"""

import numpy as np
import pandas as pd


def generate_transactions(n_rows: int = 20_000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    user_ids = rng.integers(1000, 5000, size=n_rows)

    # --- Legitimate transactions: everyday spending pattern ---
    legit_amounts = rng.gamma(shape=2.0, scale=800, size=n_legit)  # skewed, mostly small
    legit_hours = rng.normal(loc=14, scale=4, size=n_legit).clip(0, 23).astype(int)
    legit_velocity = rng.poisson(lam=1.2, size=n_legit)  # txns by this user in last hour

    # --- Fraudulent transactions: distinct pattern ---
    fraud_amounts = rng.gamma(shape=2.0, scale=4000, size=n_fraud)  # much larger amounts
    fraud_hours = rng.choice(
        [1, 2, 3, 4, 23, 0], size=n_fraud, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1]
    )  # odd hours
    fraud_velocity = rng.poisson(lam=6.0, size=n_fraud)  # rapid repeated attempts

    amounts = np.concatenate([legit_amounts, fraud_amounts])
    hours = np.concatenate([legit_hours, fraud_hours])
    velocity = np.concatenate([legit_velocity, fraud_velocity])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "amount": amounts.round(2),
            "hour_of_day": hours,
            "txn_count_last_hour": velocity,
            "is_fraud": labels.astype(int),
        }
    )

    # Shuffle rows so fraud isn't all at the bottom
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = np.arange(1, len(df) + 1)
    return df[["transaction_id", "user_id", "amount", "hour_of_day", "txn_count_last_hour", "is_fraud"]]


if __name__ == "__main__":
    df = generate_transactions()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["is_fraud"].value_counts(normalize=True))
