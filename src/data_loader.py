"""Reusable dataset loading and merge utilities for Olist analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def load_raw_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all required Olist tables with appropriate date parsing."""

    data_path = _as_path(data_dir)
    tables = {
        "orders": pd.read_csv(
            data_path / "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "payments": pd.read_csv(data_path / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(
            data_path / "olist_order_reviews_dataset.csv",
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        ),
        "customers": pd.read_csv(data_path / "olist_customers_dataset.csv"),
        "items": pd.read_csv(
            data_path / "olist_order_items_dataset.csv",
            parse_dates=["shipping_limit_date"],
        ),
        "products": pd.read_csv(data_path / "olist_products_dataset.csv"),
        "category_translation": pd.read_csv(
            data_path / "product_category_name_translation.csv"
        ),
    }
    return tables


def summarize_table_shapes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return table shapes in a compact dataframe for notebook display."""

    return pd.DataFrame(
        {
            "table": list(tables.keys()),
            "rows": [df.shape[0] for df in tables.values()],
            "columns": [df.shape[1] for df in tables.values()],
        }
    )


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payment-level rows into order-level payment features."""

    return (
        payments.sort_values("payment_value", ascending=False)
        .groupby("order_id")
        .agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "sum"),
            total_payment_value=("payment_value", "sum"),
            n_payment_methods=("payment_type", "nunique"),
        )
        .reset_index()
    )


def aggregate_items(
    items: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate item-level rows into order-level basket and category features."""

    products_en = products.merge(category_translation, on="product_category_name", how="left")
    items_agg = (
        items.merge(
            products_en[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left",
        )
        .groupby("order_id")
        .agg(
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
            product_category=("product_category_name_english", "first"),
        )
        .reset_index()
    )
    items_agg["total_order_value"] = items_agg["total_price"] + items_agg["total_freight"]
    return items_agg


def deduplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep one review per order, using the most recent review."""

    return (
        reviews.sort_values("review_creation_date", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        [["order_id", "review_score", "review_creation_date"]]
    )


def build_master_dataset(
    tables: dict[str, pd.DataFrame],
    delivered_only: bool = True,
) -> dict[str, Any]:
    """Build the order-level modeling base table from raw Olist sources."""

    orders = tables["orders"].copy()
    if delivered_only:
        orders = orders[orders["order_status"] == "delivered"].copy()

    payments_agg = aggregate_payments(tables["payments"])
    items_agg = aggregate_items(
        tables["items"],
        tables["products"],
        tables["category_translation"],
    )
    reviews_clean = deduplicate_reviews(tables["reviews"])

    master_df = (
        orders.merge(tables["customers"], on="customer_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_clean, on="order_id", how="left")
        .merge(items_agg, on="order_id", how="left")
    )

    return {
        "data": master_df,
        "orders": orders,
        "payments_agg": payments_agg,
        "items_agg": items_agg,
        "reviews_clean": reviews_clean,
    }


def summarize_join_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return null diagnostics after table joins."""

    null_count = df.isnull().sum()
    return (
        pd.DataFrame(
            {
                "column": null_count.index,
                "null_count": null_count.values,
                "null_pct": ((null_count / len(df)) * 100).round(2).values,
            }
        )
        .query("null_count > 0")
        .sort_values("null_pct", ascending=False)
        .reset_index(drop=True)
    )
