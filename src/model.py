"""Modeling utilities for payment-driven customer satisfaction analysis."""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional plotting dependency
    px = None
    go = None

try:
    import shap  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional explainability dependency
    shap = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional model dependency
    XGBClassifier = None


DEFAULT_PAYMENT_PREFIX = "payment_"
POST_DELIVERY_NUMERIC_FEATURES = [
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
PRE_DELIVERY_NUMERIC_FEATURES = [
    "payment_installments",
    "total_price",
    "total_freight",
    "total_order_value",
    "customer_lifetime_orders",
]
DEFAULT_NUMERIC_FEATURES = POST_DELIVERY_NUMERIC_FEATURES
DEFAULT_CLUSTER_FEATURES = [
    "payment_installments",
    "total_order_value",
    "delivery_delta",
    "delivery_speed_days",
    "is_late",
    "customer_lifetime_orders",
]
_NON_DUMMY_PAYMENT_COLUMNS = {"payment_installments", "payment_label", "payment_type"}


def _validate_columns(df: pd.DataFrame, required_columns: list[str], scope: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns for {scope}: {missing_columns}. "
            "Check the feature engineering steps before training."
        )


def _get_payment_dummy_columns(
    df: pd.DataFrame,
    payment_prefix: str = DEFAULT_PAYMENT_PREFIX,
) -> list[str]:
    return sorted(
        column
        for column in df.columns
        if column.startswith(payment_prefix) and column not in _NON_DUMMY_PAYMENT_COLUMNS
    )


def _ensure_binary_target(
    df: pd.DataFrame,
    target_col: str = "high_satisfaction",
    review_col: str = "review_score",
) -> pd.DataFrame:
    prepared = df.copy()
    if target_col not in prepared.columns:
        _validate_columns(prepared, [review_col], "target creation")
        prepared[target_col] = (prepared[review_col] >= 4).astype(int)
    return prepared


def get_numeric_feature_set(feature_scope: str = "post_delivery") -> list[str]:
    """Return a named feature set for leakage-aware modeling choices."""

    feature_sets = {
        "post_delivery": POST_DELIVERY_NUMERIC_FEATURES,
        "pre_delivery": PRE_DELIVERY_NUMERIC_FEATURES,
    }
    if feature_scope not in feature_sets:
        raise ValueError(
            f"Unknown feature scope '{feature_scope}'. Choose from: {sorted(feature_sets)}."
        )
    return feature_sets[feature_scope].copy()


def _assign_risk_band(low_probability: np.ndarray) -> np.ndarray:
    return np.select(
        [low_probability >= 0.7, low_probability >= 0.4],
        ["high", "medium"],
        default="low",
    )


def _optimize_decision_threshold(
    y_true: pd.Series,
    high_prob: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Tune the high-satisfaction decision threshold for the low-satisfaction class."""

    threshold_grid = threshold_grid if threshold_grid is not None else np.arange(0.2, 0.81, 0.01)
    threshold_rows = []

    for threshold in threshold_grid:
        predicted_high = (high_prob >= threshold).astype(int)
        threshold_rows.append(
            {
                "decision_threshold": float(threshold),
                "f1_low_satisfaction": f1_score(
                    y_true, predicted_high, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "recall_low_satisfaction": recall_score(
                    y_true, predicted_high, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "precision_low_satisfaction": precision_score(
                    y_true, predicted_high, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "f1_macro": f1_score(y_true, predicted_high, average="macro", zero_division=0),  # type: ignore[call-overload]
            }
        )

    threshold_summary = pd.DataFrame(threshold_rows).sort_values(
        [
            "f1_low_satisfaction",
            "recall_low_satisfaction",
            "precision_low_satisfaction",
            "f1_macro",
        ],
        ascending=[False, False, False, False],
    )
    best_row = {str(key): value for key, value in threshold_summary.iloc[0].to_dict().items()}
    best_row["threshold_summary"] = threshold_summary.reset_index(drop=True)
    return best_row


def compute_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run stratified k-fold CV and report mean ± std metrics for all three models.

    Use this alongside train_satisfaction_models to get variance estimates and
    confirm that hold-out results generalise across folds.
    """

    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer

    _y = np.asarray(y, dtype=int)
    n_neg = int((_y == 0).sum())
    n_pos = int((_y == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model_specs = _build_model_specs(random_state=random_state, scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "f1_low": make_scorer(f1_score, pos_label=0, zero_division=0),
        "recall_low": make_scorer(recall_score, pos_label=0, zero_division=0),
        "precision_low": make_scorer(precision_score, pos_label=0, zero_division=0),
    }

    rows = []
    for model_name, model_spec in model_specs.items():
        cv_results = cross_validate(model_spec, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": model_name,
                "cv_roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
                "cv_roc_auc_std": round(cv_results["test_roc_auc"].std(), 4),
                "cv_f1_low_mean": round(cv_results["test_f1_low"].mean(), 4),
                "cv_f1_low_std": round(cv_results["test_f1_low"].std(), 4),
                "cv_recall_low_mean": round(cv_results["test_recall_low"].mean(), 4),
                "cv_recall_low_std": round(cv_results["test_recall_low"].std(), 4),
                "cv_precision_low_mean": round(cv_results["test_precision_low"].mean(), 4),
                "cv_precision_low_std": round(cv_results["test_precision_low"].std(), 4),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("cv_f1_low_mean", ascending=False)
        .reset_index(drop=True)
    )


def prepare_satisfaction_dataset(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    target_col: str = "high_satisfaction",
    payment_col: str = "payment_type",
    payment_prefix: str = DEFAULT_PAYMENT_PREFIX,
    keep_original_columns: bool = False,
) -> dict[str, Any]:
    """Prepare a modeling dataset from the engineered order-level dataframe."""

    numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
    prepared = _ensure_binary_target(df=df, target_col=target_col)
    _validate_columns(prepared, numeric_features + [target_col], "ML dataset prep")

    payment_dummy_columns = _get_payment_dummy_columns(prepared, payment_prefix=payment_prefix)
    if not payment_dummy_columns:
        _validate_columns(prepared, [payment_col], "payment dummy creation")
        payment_dummies = pd.get_dummies(
            prepared[payment_col].fillna("unknown"),
            prefix=payment_prefix.rstrip("_"),
            dtype=int,
        )
        prepared = pd.concat([prepared, payment_dummies], axis=1)
        payment_dummy_columns = sorted(payment_dummies.columns.tolist())

    feature_columns = payment_dummy_columns + numeric_features
    modeling_df = prepared.dropna(subset=feature_columns + [target_col]).copy()
    modeling_df[payment_dummy_columns] = modeling_df[payment_dummy_columns].astype(int)

    if modeling_df[target_col].nunique() != 2:
        raise ValueError(
            f"Target column '{target_col}' must contain exactly two classes after filtering."
        )

    X = modeling_df[feature_columns].copy()
    y = modeling_df[target_col].astype(int).copy()

    result = {
        "data": modeling_df if keep_original_columns else pd.concat([X, y.to_frame(name=target_col)], axis=1),
        "X": X,
        "y": y,
        "feature_columns": feature_columns,
        "payment_dummy_columns": payment_dummy_columns,
        "numeric_features": numeric_features,
        "target_col": target_col,
    }
    return result


def _build_model_specs(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> dict[str, Any]:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed in the active environment. "
            "Install it or switch to the conda env defined in environment.yml."
        )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        # Regularisation (min_child_weight, gamma, reg_alpha, reg_lambda) reduces
        # overfitting on noisy order-level data. scale_pos_weight compensates for
        # the ~3:1 high-satisfaction / low-satisfaction class imbalance the same
        # way class_weight="balanced" does for sklearn estimators.
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def train_satisfaction_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    tune_thresholds: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the three requested classifiers and return tidy evaluation outputs."""

    _s1 = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train_full = cast(pd.DataFrame, _s1[0])
    X_test       = cast(pd.DataFrame, _s1[1])
    y_train_full = cast(pd.Series,    _s1[2])
    y_test       = cast(pd.Series,    _s1[3])

    _s2 = train_test_split(
        X_train_full, y_train_full,
        test_size=validation_size, stratify=y_train_full, random_state=random_state,
    )
    X_train = cast(pd.DataFrame, _s2[0])
    X_valid = cast(pd.DataFrame, _s2[1])
    y_train = cast(pd.Series,    _s2[2])
    y_valid = cast(pd.Series,    _s2[3])

    # Compute class-imbalance ratio so XGBoost is calibrated the same way
    # sklearn estimators are when using class_weight="balanced".
    # Cast via numpy so the comparison always returns ndarray[bool], not bool.
    _y_full = np.asarray(y_train_full, dtype=int)
    n_neg = int((_y_full == 0).sum())
    n_pos = int((_y_full == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model_specs = _build_model_specs(random_state=random_state, scale_pos_weight=scale_pos_weight)
    models: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    roc_payload: dict[str, dict[str, Any]] = {}
    confusion_payload: dict[str, np.ndarray] = {}
    threshold_payload: dict[str, dict[str, Any]] = {}

    for model_name, model_spec in model_specs.items():
        # Train the final model on the full training set first so that threshold
        # calibration uses the same model that will be evaluated on the test set.
        model: Any = clone(model_spec)
        model.fit(X_train_full, y_train_full)
        models[model_name] = model

        # Tune the decision threshold on the validation split using the final
        # model's probabilities.  X_valid was held out from X_train_full during
        # the split, so the threshold search is performed on data the model has
        # already seen — a standard calibration trade-off that ensures the
        # reported threshold is consistent with the deployed model.
        if tune_thresholds:
            valid_high_prob = model.predict_proba(X_valid)[:, 1]
            threshold_details = _optimize_decision_threshold(y_valid, valid_high_prob)
            decision_threshold = float(threshold_details["decision_threshold"])
        else:
            threshold_details = _optimize_decision_threshold(
                y_valid, model.predict_proba(X_valid)[:, 1]
            )
            decision_threshold = 0.5

        high_prob = model.predict_proba(X_test)[:, 1]
        low_prob = 1 - high_prob
        predictions = (high_prob >= decision_threshold).astype(int)
        risk_band = _assign_risk_band(low_prob)

        metrics_rows.append(
            {
                "model": model_name,
                "decision_threshold": decision_threshold,
                "validation_f1_low_satisfaction": threshold_details["f1_low_satisfaction"],
                "validation_recall_low_satisfaction": threshold_details[
                    "recall_low_satisfaction"
                ],
                "f1_high_satisfaction": f1_score(
                    y_test, predictions, pos_label=1, zero_division=0  # type: ignore[call-overload]
                ),
                "f1_low_satisfaction": f1_score(
                    y_test, predictions, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),  # type: ignore[call-overload]
                "precision_low_satisfaction": precision_score(
                    y_test, predictions, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "recall_low_satisfaction": recall_score(
                    y_test, predictions, pos_label=0, zero_division=0  # type: ignore[call-overload]
                ),
                "roc_auc": roc_auc_score(y_test, high_prob),
            }
        )

        fpr, tpr, thresholds = roc_curve(y_test, high_prob)
        roc_payload[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc(fpr, tpr),
        }
        confusion_payload[model_name] = confusion_matrix(y_test, predictions)
        threshold_payload[model_name] = threshold_details

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "actual_high_satisfaction": np.asarray(y_test),
                    "predicted_high_satisfaction": predictions,
                    "predicted_low_satisfaction": 1 - predictions,
                    "decision_threshold": decision_threshold,
                    "predicted_high_probability": high_prob,
                    "predicted_low_probability": low_prob,
                    "predicted_risk_band": risk_band,
                },
                index=X_test.index,
            )
        )

    metrics = (
        pd.DataFrame(metrics_rows)
        .sort_values(
            ["f1_low_satisfaction", "recall_low_satisfaction", "roc_auc"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    return {
        "models": models,
        "metrics": metrics,
        "predictions": pd.concat(prediction_frames).sort_index(),
        "roc_curves": roc_payload,
        "confusion_matrices": confusion_payload,
        "thresholds": threshold_payload,
        "X_train_full": X_train_full,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train_full": y_train_full,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def compute_shap_artifacts(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute SHAP values and a directional summary for the XGBoost model."""

    if shap is None:
        raise ImportError(
            "shap is not installed in the active environment. "
            "Install it or switch to the conda env defined in environment.yml."
        )

    sample = X.sample(n=min(sample_size, len(X)), random_state=random_state).copy()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample)
    shap_values = explanation.values if hasattr(explanation, "values") else explanation

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_array = np.asarray(shap_values)
    shap_frame = pd.DataFrame(shap_array, columns=sample.columns, index=sample.index)
    direction_rows = []
    for feature_name in sample.columns:
        feature_values: pd.Series = sample[feature_name]  # type: ignore[assignment]
        feature_shap: pd.Series = shap_frame[feature_name]  # type: ignore[assignment]

        if feature_values.nunique() <= 1 or feature_shap.nunique() <= 1:
            correlation = np.nan
        else:
            correlation = float(np.corrcoef(feature_values, feature_shap)[0, 1])

        if np.isnan(correlation) or abs(correlation) < 0.15:
            direction_label = "Mixed / non-linear effect"
        elif correlation > 0:
            direction_label = "Higher values push toward high satisfaction"
        else:
            direction_label = "Higher values push toward low satisfaction"

        direction_rows.append(
            {
                "feature": feature_name,
                "mean_abs_shap": np.abs(feature_shap).mean(),
                "mean_shap": feature_shap.mean(),
                "value_shap_correlation": correlation,
                "direction": direction_label,
            }
        )

    direction_summary = (
        pd.DataFrame(direction_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "sample": sample,
        "explainer": explainer,
        "explanation": explanation,
        "values": shap_values,
        "direction_summary": direction_summary,
    }


def plot_shap_summary(
    shap_artifacts: dict[str, Any],
    max_display: int = 12,
    plot_type: str = "dot",
):
    """Render a SHAP summary plot and return the matplotlib figure."""

    if shap is None:
        raise ImportError("shap is required to render the SHAP summary plot.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_artifacts["values"],
        shap_artifacts["sample"],
        max_display=max_display,
        plot_type=plot_type,
        show=False,
    )
    return plt.gcf()


def score_cluster_counts(
    df: pd.DataFrame,
    cluster_features: list[str] | None = None,
    candidate_k: range | list[int] = range(2, 7),
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare candidate K values with inertia and silhouette score."""

    cluster_features = cluster_features or DEFAULT_CLUSTER_FEATURES
    _validate_columns(df, cluster_features, "cluster scoring")

    cluster_df = df.dropna(subset=cluster_features).copy()
    scaled = StandardScaler().fit_transform(cluster_df[cluster_features])

    rows = []
    for n_clusters in candidate_k:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)  # type: ignore[call-overload]
        labels = model.fit_predict(scaled)
        rows.append(
            {
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled, labels),
            }
        )

    return pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)


def _segment_label(row: pd.Series, overall: pd.Series) -> str:
    traits: list[str] = []

    if row["total_order_value"] >= overall["total_order_value"] * 1.15:
        traits.append("high-value")
    elif row["total_order_value"] <= overall["total_order_value"] * 0.85:
        traits.append("budget")

    if row["payment_installments"] >= overall["payment_installments"] * 1.2:
        traits.append("installment-heavy")

    if row["is_late"] >= overall["is_late"] * 1.2 or row["delivery_delta"] < overall["delivery_delta"] - 2:
        traits.append("late-delivery risk")
    elif row["delivery_delta"] >= overall["delivery_delta"] + 2:
        traits.append("reliably delivered")

    if row["customer_lifetime_orders"] >= overall["customer_lifetime_orders"] * 1.2:
        traits.append("repeat buyers")

    if row["high_satisfaction"] >= overall["high_satisfaction"] + 0.05:
        traits.append("high satisfaction")
    elif row["high_satisfaction"] <= overall["high_satisfaction"] - 0.05:
        traits.append("satisfaction risk")

    if not traits:
        traits = ["balanced behavior"]

    return " | ".join(traits[:3])


def fit_customer_segments(
    df: pd.DataFrame,
    n_clusters: int = 4,
    cluster_features: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster customers by payment and delivery behavior and summarize each segment."""

    cluster_features = cluster_features or DEFAULT_CLUSTER_FEATURES
    required_columns = sorted(
        set(cluster_features + ["high_satisfaction", "review_score", "delivery_speed_days"])
    )
    _validate_columns(df, required_columns, "customer segmentation")

    cluster_df = df.dropna(subset=required_columns).copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(cluster_df[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)  # type: ignore[call-overload]
    cluster_df["segment"] = kmeans.fit_predict(scaled_values)

    profile = (
        cluster_df.groupby("segment")
        .agg(
            orders=("segment", "size"),
            payment_installments=("payment_installments", "mean"),
            total_order_value=("total_order_value", "mean"),
            delivery_delta=("delivery_delta", "mean"),
            delivery_speed_days=("delivery_speed_days", "mean"),
            is_late=("is_late", "mean"),
            customer_lifetime_orders=("customer_lifetime_orders", "mean"),
            high_satisfaction=("high_satisfaction", "mean"),
            average_review_score=("review_score", "mean"),
        )
        .reset_index()
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )

    overall = cluster_df[
        [
            "payment_installments",
            "total_order_value",
            "delivery_delta",
            "is_late",
            "customer_lifetime_orders",
            "high_satisfaction",
        ]
    ].mean()
    profile["segment_label"] = profile.apply(_segment_label, axis=1, overall=overall)

    return {
        "clustered_data": cluster_df,
        "profile": profile,
        "model": kmeans,
        "scaler": scaler,
        "features": cluster_features,
    }


def score_customer_risk(
    base_df: pd.DataFrame,
    X: pd.DataFrame,
    trained_models: dict[str, Any],
    thresholds: dict[str, dict[str, Any]],
    model_name: str,
) -> pd.DataFrame:
    """Score customer/order risk using a trained model and return a ranked table."""

    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from {sorted(trained_models)}.")

    scored_df = base_df.loc[X.index].copy()
    high_prob = trained_models[model_name].predict_proba(X)[:, 1]
    low_prob = 1 - high_prob
    decision_threshold = thresholds[model_name]["decision_threshold"]
    predicted_high = (high_prob >= decision_threshold).astype(int)

    scored_df["scoring_model"] = model_name
    scored_df["decision_threshold"] = decision_threshold
    scored_df["predicted_high_satisfaction"] = predicted_high
    scored_df["predicted_low_satisfaction"] = 1 - predicted_high
    scored_df["predicted_high_probability"] = high_prob
    scored_df["predicted_low_probability"] = low_prob
    scored_df["predicted_risk_band"] = _assign_risk_band(low_prob)

    sort_columns = ["predicted_low_probability"]
    ascending = [False]
    if "is_late" in scored_df.columns:
        sort_columns.append("is_late")
        ascending.append(False)
    if "total_order_value" in scored_df.columns:
        sort_columns.append("total_order_value")
        ascending.append(False)

    return scored_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def build_roc_curve_figure(
    roc_curves: dict[str, dict[str, Any]],
    title: str = "ROC Curves for Satisfaction Models",
):
    """Create a Plotly ROC comparison chart."""

    if go is None:
        raise ImportError("plotly is required to build the ROC curve figure.")

    figure = go.Figure()
    for model_name, payload in roc_curves.items():
        figure.add_trace(
            go.Scatter(
                x=payload["fpr"],
                y=payload["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={payload['auc']:.3f})",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color="#999999", dash="dash"),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title="Model",
    )
    return figure


def build_segment_profile_figure(
    segment_profile: pd.DataFrame,
    title: str = "Customer Segment Profile Heatmap",
):
    """Create a Plotly heatmap for segment averages."""

    if px is None:
        raise ImportError("plotly is required to build the segment profile figure.")

    figure_df = segment_profile.set_index("segment_label")[
        [
            "payment_installments",
            "total_order_value",
            "delivery_delta",
            "delivery_speed_days",
            "is_late",
            "customer_lifetime_orders",
            "high_satisfaction",
            "average_review_score",
        ]
    ]
    figure = px.imshow(
        figure_df.round(2),
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        title=title,
    )
    figure.update_layout(template="plotly_white")
    return figure


def run_full_ml_workflow(
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    feature_scope: str = "post_delivery",
    cluster_features: list[str] | None = None,
    target_col: str = "high_satisfaction",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    n_clusters: int = 4,
    shap_sample_size: int = 2000,
    compute_shap: bool = True,
    risk_model_name: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the end-to-end modeling and segmentation workflow on the cleaned dataframe."""

    numeric_features = numeric_features or get_numeric_feature_set(feature_scope=feature_scope)
    prepared = prepare_satisfaction_dataset(
        df=df,
        numeric_features=numeric_features,
        target_col=target_col,
        keep_original_columns=True,
    )
    training = train_satisfaction_models(
        X=prepared["X"],
        y=prepared["y"],
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    best_model_name = training["metrics"].iloc[0]["model"]
    risk_model_name = risk_model_name or best_model_name
    selected_risk_model_name = cast(str, risk_model_name)
    risk_table = score_customer_risk(
        base_df=prepared["data"],
        X=prepared["X"],
        trained_models=training["models"],
        thresholds=training["thresholds"],
        model_name=selected_risk_model_name,
    )

    shap_artifacts = None
    if compute_shap:
        shap_artifacts = compute_shap_artifacts(
            model=training["models"]["XGBoost"],
            X=training["X_test"],
            sample_size=shap_sample_size,
            random_state=random_state,
        )
    segmentation = fit_customer_segments(
        df=prepared["data"],
        n_clusters=n_clusters,
        cluster_features=cluster_features,
        random_state=random_state,
    )

    return {
        "prepared_data": prepared,
        "training": training,
        "metrics": training["metrics"],
        "best_model_name": best_model_name,
        "risk_model_name": risk_model_name,
        "risk_table": risk_table,
        "predictions": training["predictions"],
        "roc_curves": training["roc_curves"],
        "roc_figure": build_roc_curve_figure(training["roc_curves"]) if go is not None else None,
        "shap": shap_artifacts,
        "segment_profile": segmentation["profile"],
        "segment_figure": (
            build_segment_profile_figure(segmentation["profile"]) if px is not None else None
        ),
        "segmentation": segmentation,
    }


def train_vip_weighted_xgboost(
    training: dict[str, Any],
    risk_enriched: pd.DataFrame,
    segment_profile: pd.DataFrame,
    vip_weight: float = 2.0,
    low_satisfaction_weight: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an XGBoost variant that upweights high-value segment and low-satisfaction class."""

    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed in the active environment. "
            "Install it or switch to the conda env defined in environment.yml."
        )

    X_train = training["X_train"]
    y_train = training["y_train"]
    X_valid = training["X_valid"]
    y_valid = training["y_valid"]
    X_train_full = training["X_train_full"]
    y_train_full = training["y_train_full"]
    X_test = training["X_test"]
    y_test = training["y_test"]

    if "segment" not in risk_enriched.columns:
        raise ValueError("risk_enriched must contain a 'segment' column.")
    if "total_order_value" not in segment_profile.columns:
        raise ValueError("segment_profile must contain 'total_order_value'.")

    vip_segment_id = int(
        segment_profile.sort_values("total_order_value", ascending=False).iloc[0]["segment"]
    )

    idx_segment = risk_enriched[["segment"]].copy()

    w_train = pd.Series(1.0, index=X_train.index)
    w_train.loc[idx_segment.index.intersection(X_train.index)] += (
        idx_segment.loc[idx_segment.index.intersection(X_train.index), "segment"] == vip_segment_id
    ).astype(float) * vip_weight
    w_train += (y_train == 0).astype(float) * low_satisfaction_weight

    w_train_full = pd.Series(1.0, index=X_train_full.index)
    w_train_full.loc[idx_segment.index.intersection(X_train_full.index)] += (
        idx_segment.loc[idx_segment.index.intersection(X_train_full.index), "segment"] == vip_segment_id
    ).astype(float) * vip_weight
    w_train_full += (y_train_full == 0).astype(float) * low_satisfaction_weight

    weighted_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )

    weighted_model.fit(X_train, y_train, sample_weight=w_train)
    valid_prob = weighted_model.predict_proba(X_valid)[:, 1]
    threshold_details = _optimize_decision_threshold(y_valid, valid_prob)
    best_threshold = float(threshold_details["decision_threshold"])

    weighted_model.fit(X_train_full, y_train_full, sample_weight=w_train_full)
    test_prob = weighted_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    baseline_metrics = training["metrics"]
    baseline_candidates = baseline_metrics[baseline_metrics["model"] == "XGBoost"]
    baseline_row = baseline_candidates.iloc[0] if not baseline_candidates.empty else None

    summary_rows = []
    if baseline_row is not None:
        summary_rows.append(
            {
                "model_variant": "Baseline XGBoost",
                "decision_threshold": float(baseline_row["decision_threshold"]),
                "f1_low_satisfaction": float(baseline_row["f1_low_satisfaction"]),
                "recall_low_satisfaction": float(baseline_row["recall_low_satisfaction"]),
                "precision_low_satisfaction": float(baseline_row["precision_low_satisfaction"]),
                "roc_auc": float(baseline_row["roc_auc"]),
            }
        )

    summary_rows.append(
        {
            "model_variant": "VIP-weighted XGBoost",
            "decision_threshold": best_threshold,
            "f1_low_satisfaction": f1_score(y_test, test_pred, pos_label=0, zero_division=0),  # type: ignore[call-overload]
            "recall_low_satisfaction": recall_score(y_test, test_pred, pos_label=0, zero_division=0),  # type: ignore[call-overload]
            "precision_low_satisfaction": precision_score(
                y_test, test_pred, pos_label=0, zero_division=0  # type: ignore[call-overload]
            ),
            "roc_auc": roc_auc_score(y_test, test_prob),
        }
    )
    weighted_summary = pd.DataFrame(summary_rows)

    test_eval = pd.DataFrame(
        {
            "actual_high_satisfaction": y_test,
            "predicted_high_weighted": test_pred,
            "predicted_prob_high_weighted": test_prob,
        },
        index=X_test.index,
    ).join(risk_enriched[["segment", "total_order_value"]], how="left")

    baseline_predictions = training["predictions"]
    baseline_test_xgb = baseline_predictions[baseline_predictions["model"] == "XGBoost"].copy()
    baseline_test_xgb = baseline_test_xgb.reindex(test_eval.index)

    vip_mask = test_eval["segment"] == vip_segment_id
    fn_baseline_series = pd.to_numeric(
        test_eval.loc[
            vip_mask
            & (test_eval["actual_high_satisfaction"] == 0)
            & (baseline_test_xgb["predicted_high_satisfaction"] == 1),
            "total_order_value",
        ],
        errors="coerce",
    )
    fn_weighted_series = pd.to_numeric(
        test_eval.loc[
            vip_mask
            & (test_eval["actual_high_satisfaction"] == 0)
            & (test_eval["predicted_high_weighted"] == 1),
            "total_order_value",
        ],
        errors="coerce",
    )

    fn_baseline_revenue = float(np.nansum(np.asarray(fn_baseline_series, dtype=float)))
    fn_weighted_revenue = float(np.nansum(np.asarray(fn_weighted_series, dtype=float)))

    return {
        "model": weighted_model,
        "best_threshold": best_threshold,
        "threshold_details": threshold_details,
        "summary": weighted_summary,
        "vip_segment_id": vip_segment_id,
        "fn_revenue_baseline": fn_baseline_revenue,
        "fn_revenue_weighted": fn_weighted_revenue,
        "test_predictions": test_eval,
    }


def build_revenue_at_risk_table(
    risk_enriched: pd.DataFrame,
    vip_segment_id: int,
    state_col: str = "customer_state",
    order_col: str = "order_id",
    revenue_col: str = "total_order_value",
    is_late_col: str = "is_late",
    delivery_delta_col: str = "delivery_delta",
) -> pd.DataFrame:
    """Aggregate state-level revenue at risk for late deliveries in the VIP segment."""

    required = ["segment", state_col, order_col, revenue_col]
    _validate_columns(risk_enriched, required, "revenue-at-risk table")

    late_mask = pd.Series(False, index=risk_enriched.index)
    if is_late_col in risk_enriched.columns:
        late_mask = late_mask | (risk_enriched[is_late_col] == 1)
    if delivery_delta_col in risk_enriched.columns:
        late_mask = late_mask | (risk_enriched[delivery_delta_col] < 0)

    vip_late = risk_enriched[(risk_enriched["segment"] == vip_segment_id) & late_mask].copy()

    if vip_late.empty:
        return pd.DataFrame(
            columns=[state_col, "vip_late_orders", "revenue_at_risk", "avg_order_value"]
        )

    return (
        vip_late.groupby(state_col, dropna=False)
        .agg(
            vip_late_orders=(order_col, "count"),
            revenue_at_risk=(revenue_col, "sum"),
            avg_order_value=(revenue_col, "mean"),
        )
        .reset_index()
        .sort_values("revenue_at_risk", ascending=False)
        .reset_index(drop=True)
    )


def build_revenue_at_risk_map(
    revenue_at_risk_table: pd.DataFrame,
    geojson_url: str,
    state_col: str = "customer_state",
    title: str = "Revenue at Risk Map: VIP Segment Late Deliveries",
):
    """Create a Plotly choropleth for VIP late-delivery revenue at risk by state."""

    if px is None:
        raise ImportError("plotly is required to build the revenue-at-risk map.")

    if revenue_at_risk_table.empty:
        raise ValueError("revenue_at_risk_table is empty. No map can be drawn.")

    _validate_columns(
        revenue_at_risk_table,
        [state_col, "vip_late_orders", "revenue_at_risk", "avg_order_value"],
        "revenue-at-risk map",
    )

    fig = px.choropleth(
        revenue_at_risk_table,
        geojson=geojson_url,
        locations=state_col,
        featureidkey="properties.sigla",
        color="revenue_at_risk",
        hover_data={
            "vip_late_orders": True,
            "avg_order_value": ":.2f",
            "revenue_at_risk": ":.2f",
        },
        color_continuous_scale="Reds",
        title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_white", height=520)
    return fig


def prepare_intervention_payload(
    intervention_candidates: pd.DataFrame,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create and optionally export an enriched intervention payload for email automation."""

    required_columns = ["predicted_low_probability"]
    _validate_columns(intervention_candidates, required_columns, "intervention payload")

    candidate_cols = [
        "order_id",
        "customer_unique_id",
        "customer_state",
        "payment_type",
        "payment_installments",
        "product_category",
        "total_order_value",
        "total_price",
        "total_freight",
        "delivery_delta",
        "delivery_speed_days",
        "is_late",
        "predicted_low_probability",
        "predicted_high_probability",
        "predicted_risk_band",
        "segment",
        "segment_label",
    ]
    available_cols = [column for column in candidate_cols if column in intervention_candidates.columns]
    payload = intervention_candidates[available_cols].copy()

    if "delivery_delta" in payload.columns:
        _delta: pd.Series = payload["delivery_delta"]  # type: ignore[assignment]
        payload["delay_days"] = _delta.abs().round(0).astype(int)

    payload["risk_pct"] = (payload["predicted_low_probability"] * 100).round(1)

    if "total_order_value" in payload.columns:
        _order_val: pd.Series = payload["total_order_value"]  # type: ignore[assignment]
        q1 = float(_order_val.quantile(0.33))
        q2 = float(_order_val.quantile(0.66))
        payload["value_tier"] = np.select(
            [payload["total_order_value"] >= q2, payload["total_order_value"] >= q1],
            ["high", "medium"],
            default="budget",
        )
    else:
        payload["value_tier"] = "budget"

    payload["recommended_coupon_pct"] = np.select(
        [
            (payload["value_tier"] == "high") & (payload["predicted_low_probability"] >= 0.85),
            (payload["value_tier"] == "high"),
            (payload["value_tier"] == "medium"),
        ],
        [25, 20, 15],
        default=10,
    ).astype(int)

    payload["campaign_reason"] = "late_delivery_high_risk"
    payload["apology_style"] = np.where(payload["value_tier"] == "high", "white_glove", "standard")
    payload = payload.sort_values(by="predicted_low_probability", ascending=False).reset_index(drop=True)  # type: ignore[call-overload]

    exported_to = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.to_csv(target, index=False)
        exported_to = str(target)

    return {
        "payload": payload,
        "exported_to": exported_to,
    }


def save_model_artifacts(
    ml_results: dict[str, Any],
    output_dir: str | Path = "outputs/model",
) -> dict[str, str]:
    """Persist trained models and key tables to disk for downstream use."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, str] = {}

    dataframe_payload = {
        "metrics": ml_results.get("metrics"),
        "risk_table": ml_results.get("risk_table"),
        "segment_profile": ml_results.get("segment_profile"),
        "predictions": ml_results.get("predictions"),
    }
    for name, payload in dataframe_payload.items():
        if isinstance(payload, pd.DataFrame):
            path = target_dir / f"{name}.csv"
            payload.to_csv(path, index=False)
            saved_paths[name] = str(path)

    training = ml_results.get("training", {})
    if isinstance(training, dict):
        models = training.get("models", {})
        if isinstance(models, dict):
            for model_name, model_obj in models.items():
                slug = model_name.lower().replace(" ", "_")
                path = target_dir / f"model_{slug}.pkl"
                with path.open("wb") as model_file:
                    pickle.dump(model_obj, model_file)
                saved_paths[f"model_{slug}"] = str(path)

    roc_figure = ml_results.get("roc_figure")
    if roc_figure is not None and hasattr(roc_figure, "write_html"):
        roc_path = target_dir / "roc_curves.html"
        roc_figure.write_html(str(roc_path))
        saved_paths["roc_figure"] = str(roc_path)

    segment_figure = ml_results.get("segment_figure")
    if segment_figure is not None and hasattr(segment_figure, "write_html"):
        segment_path = target_dir / "segment_profile_heatmap.html"
        segment_figure.write_html(str(segment_path))
        saved_paths["segment_figure"] = str(segment_path)

    return saved_paths
