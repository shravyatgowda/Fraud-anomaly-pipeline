"""
train.py

Trains a Random Forest classifier on the synthetic transaction dataset
and reports precision, recall, F1, and ROC-AUC on a held-out test set.

Usage:
    python -m src.train
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import build_feature_vector, compute_amount_deviation, is_odd_hour

FEATURE_COLUMNS = ["amount", "amount_deviation", "hour_of_day", "is_odd_hour", "txn_count_last_hour"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature engineering over the full training set."""
    df = df.copy()
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).fillna(0)
    df = df.join(user_stats, on="user_id", rsuffix="_user")
    df["amount_deviation"] = df.apply(
        lambda r: compute_amount_deviation(r["amount"], r["mean"], r["std"]), axis=1
    )
    df["is_odd_hour"] = df["hour_of_day"].apply(lambda h: int(is_odd_hour(h)))
    return df


def main():
    data_path = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python data/generate_synthetic_data.py` first."
        )

    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    model_path = Path(__file__).resolve().parent.parent / "model.joblib"
    joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
