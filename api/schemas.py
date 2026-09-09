from pydantic import BaseModel


class TransactionIn(BaseModel):
    transaction_id: int
    user_id: int
    amount: float
    hour_of_day: int


class ScoreOut(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_flagged: bool
    features_used: dict
    latency_ms: float
