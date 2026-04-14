"""Feature engineering helpers for retention and satisfaction modeling."""

from __future__ import annotations

from typing import Any

import pandas as pd

PAYMENT_LABELS = {
    "credit_card": "Credit card",
    "boleto": "Boleto",
    "voucher": "Voucher",
    "debit_card": "Debit card",
}

DEFAULT_ML_FEATURES = [
    "payment_installments",
    "total_price",
    "total_freight",
    "delivery_delta",
    "delivery_speed_days",
    "is_late",
    "total_order_value",
    "customer_lifetime_orders",
    "review_sentiment",
]


def engineer_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["delivery_delta"] = (
        enriched["order_estimated_delivery_date"] - enriched["order_delivered_customer_date"]
    ).dt.days
    enriched["delivery_speed_days"] = (
        enriched["order_delivered_customer_date"] - enriched["order_purchase_timestamp"]
    ).dt.days
    enriched["is_late"] = (enriched["delivery_delta"] < 0).astype(int)
    return enriched


def add_customer_lifetime_orders(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    customer_order_counts = (
        enriched.groupby("customer_unique_id")["order_id"]
        .count()
        .reset_index()
        .rename(columns={"order_id": "customer_lifetime_orders"})
    )
    return enriched.merge(customer_order_counts, on="customer_unique_id", how="left")


def add_payment_dummies(df: pd.DataFrame, prefix: str = "payment") -> pd.DataFrame:
    enriched = df.copy()
    payment_dummies = pd.get_dummies(enriched["payment_type"], prefix=prefix, dtype=int)
    return pd.concat([enriched, payment_dummies], axis=1)


def add_satisfaction_target(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.dropna(subset=["review_score"]).copy()
    enriched["high_satisfaction"] = (enriched["review_score"] >= 4).astype(int)
    return enriched


def add_eda_time_and_labels(
    df: pd.DataFrame,
    payment_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    enriched = df.copy()
    label_map = payment_labels or PAYMENT_LABELS
    enriched["payment_label"] = enriched["payment_type"].map(label_map).fillna("Other")
    enriched["year"] = enriched["order_purchase_timestamp"].dt.year
    enriched["year_month"] = enriched["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return enriched


def add_review_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TextBlob sentiment polarity from review_comment_message.

    Fills 0.0 for rows where the message is null or empty.
    Falls back to 0.0 for all rows if textblob is not installed.
    """
    try:
        from textblob import TextBlob  # type: ignore[import-not-found]
    except ImportError:
        enriched = df.copy()
        enriched["review_sentiment"] = 0.0
        return enriched

    enriched = df.copy()
    texts: pd.Series = enriched.get("review_comment_message", pd.Series(dtype=str)).fillna("")  # type: ignore[assignment]
    enriched["review_sentiment"] = texts.map(
        lambda t: TextBlob(str(t)).sentiment.polarity if t else 0.0
    )
    return enriched


def build_feature_engineered_dataset(
    df: pd.DataFrame,
    add_dummies: bool = True,
    add_target: bool = True,
) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in a deterministic order."""

    engineered = engineer_delivery_features(df)
    engineered = add_customer_lifetime_orders(engineered)
    if add_dummies:
        engineered = add_payment_dummies(engineered, prefix="payment")
    if add_target:
        engineered = add_satisfaction_target(engineered)
    engineered = add_eda_time_and_labels(engineered)
    engineered = add_review_sentiment(engineered)
    return engineered


def build_clean_dataset(
    df: pd.DataFrame,
    ml_features: list[str] | None = None,
) -> dict[str, Any]:
    """Return cleaned modeling table plus diagnostics."""

    required_features = ml_features or DEFAULT_ML_FEATURES
    clean_df = df.dropna(subset=required_features).copy()
    return {
        "data": clean_df,
        "ml_features": required_features,
        "dropped_rows": int(len(df) - len(clean_df)),
    }
