"""Utilities for the Olist payment analytics project."""

from .data_loader import (
    aggregate_items,
    aggregate_payments,
    build_master_dataset,
    deduplicate_reviews,
    load_raw_tables,
    summarize_join_quality,
    summarize_table_shapes,
)
from .feature_engineering import (
    DEFAULT_ML_FEATURES,
    PAYMENT_LABELS,
    add_customer_lifetime_orders,
    add_eda_time_and_labels,
    add_payment_dummies,
    add_satisfaction_target,
    build_clean_dataset,
    build_feature_engineered_dataset,
    engineer_delivery_features,
)

_MODEL_EXPORTS_AVAILABLE = True
try:
    from .model import (
        DEFAULT_CLUSTER_FEATURES,
        DEFAULT_NUMERIC_FEATURES,
        DEFAULT_PAYMENT_PREFIX,
        POST_DELIVERY_NUMERIC_FEATURES,
        PRE_DELIVERY_NUMERIC_FEATURES,
        build_roc_curve_figure,
        build_segment_profile_figure,
        build_revenue_at_risk_map,
        build_revenue_at_risk_table,
        compute_cv_metrics,
        compute_shap_artifacts,
        fit_customer_segments,
        get_numeric_feature_set,
        plot_shap_summary,
        prepare_satisfaction_dataset,
        save_model_artifacts,
        run_full_ml_workflow,
        score_customer_risk,
        score_cluster_counts,
        prepare_intervention_payload,
        train_vip_weighted_xgboost,
        train_satisfaction_models,
    )
except ModuleNotFoundError:
    _MODEL_EXPORTS_AVAILABLE = False

__all__ = [
    "aggregate_items",
    "aggregate_payments",
    "build_clean_dataset",
    "build_feature_engineered_dataset",
    "build_master_dataset",
    "deduplicate_reviews",
    "DEFAULT_ML_FEATURES",
    "PAYMENT_LABELS",
    "add_customer_lifetime_orders",
    "add_eda_time_and_labels",
    "add_payment_dummies",
    "add_satisfaction_target",
    "engineer_delivery_features",
    "load_raw_tables",
    "summarize_join_quality",
    "summarize_table_shapes",
]

if _MODEL_EXPORTS_AVAILABLE:
    __all__.extend(
        [
            "DEFAULT_CLUSTER_FEATURES",
            "DEFAULT_NUMERIC_FEATURES",
            "DEFAULT_PAYMENT_PREFIX",
            "POST_DELIVERY_NUMERIC_FEATURES",
            "PRE_DELIVERY_NUMERIC_FEATURES",
            "build_roc_curve_figure",
            "build_segment_profile_figure",
            "build_revenue_at_risk_map",
            "build_revenue_at_risk_table",
            "compute_cv_metrics",
            "compute_shap_artifacts",
            "fit_customer_segments",
            "get_numeric_feature_set",
            "plot_shap_summary",
            "prepare_satisfaction_dataset",
            "save_model_artifacts",
            "run_full_ml_workflow",
            "score_customer_risk",
            "score_cluster_counts",
            "prepare_intervention_payload",
            "train_vip_weighted_xgboost",
            "train_satisfaction_models",
        ]
    )
