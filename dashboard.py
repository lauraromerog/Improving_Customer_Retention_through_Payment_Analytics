"""Interactive EDA dashboard for Olist payment analytics."""

from __future__ import annotations

from pathlib import Path

import dash
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from src.data_loader import build_master_dataset, load_raw_tables
from src.feature_engineering import build_feature_engineered_dataset


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLOR_MAP = {
    "Credit card": "#185FA5",
    "Boleto":       "#D85A30",
    "Voucher":      "#1D9E75",
    "Debit card":   "#7F77DD",
}

BRAND_BLUE   = "#185FA5"
BRAND_RED    = "#E05252"
BRAND_GREEN  = "#1D9E75"
BRAND_ORANGE = "#D85A30"
BG_PAGE      = "#F0F2F6"
BG_CARD      = "#FFFFFF"
TEXT_PRIMARY = "#1A1A2E"
TEXT_MUTED   = "#7A7A8C"

CHART_LAYOUT = dict(
    plot_bgcolor=BG_CARD,
    paper_bgcolor=BG_CARD,
    font_family="'Inter', 'Segoe UI', sans-serif",
    font_color=TEXT_PRIMARY,
    font_size=12,
    margin=dict(l=12, r=12, t=44, b=12),
)

BRAZIL_GEOJSON = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood"
    "/master/public/data/brazil-states.geojson"
)

# Tab appearance
TAB_STYLE = {
    "padding": "10px 22px",
    "color": TEXT_MUTED,
    "borderBottom": "3px solid transparent",
    "background": "transparent",
    "fontWeight": "500",
    "fontSize": "13px",
}
SELECTED_TAB_STYLE = {
    **TAB_STYLE,
    "color": BRAND_BLUE,
    "borderBottom": f"3px solid {BRAND_BLUE}",
    "fontWeight": "700",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dashboard_data() -> pd.DataFrame:
    """Load and prepare order-level data for dashboard exploration."""
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    tables = load_raw_tables(Path(path))
    master_payload = build_master_dataset(tables=tables, delivered_only=True)
    df = build_feature_engineered_dataset(
        master_payload["data"],
        add_dummies=False,
        add_target=True,
    )
    return df.dropna(
        subset=["payment_installments", "total_order_value", "review_score"]
    ).copy()


def try_load_model_outputs() -> dict[str, pd.DataFrame]:
    """Load saved ML outputs from disk if available (non-blocking)."""
    outputs_dir = Path(__file__).parent / "outputs" / "model"
    result: dict[str, pd.DataFrame] = {}
    for name in ("metrics", "risk_table", "segment_profile"):
        path = outputs_dir / f"{name}.csv"
        if path.exists():
            try:
                result[name] = pd.read_csv(path)
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def card(label: str, value: str, sub: str = "", accent: str = BRAND_BLUE):
    """KPI card with a coloured left-border accent."""
    return html.Div(
        style={
            "background": BG_CARD,
            "borderRadius": "10px",
            "padding": "16px 20px",
            "borderLeft": f"4px solid {accent}",
            "flex": "1",
            "minWidth": "120px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.07)",
        },
        children=[
            html.Div(
                label,
                style={
                    "fontSize": "10px",
                    "color": TEXT_MUTED,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.7px",
                    "marginBottom": "6px",
                },
            ),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": TEXT_PRIMARY}),
            html.Div(sub, style={"fontSize": "10px", "color": "#BBBBCC", "marginTop": "4px"}),
        ],
    )


def chart_card(figure, span_full: bool = False, height: int | None = None):
    """Wrap a Plotly figure in a styled card."""
    style: dict = {
        "background": BG_CARD,
        "borderRadius": "12px",
        "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
        "padding": "12px",
        "width": "100%",
        "boxSizing": "border-box",
    }
    if span_full:
        style["gridColumn"] = "1 / -1"
    graph_style: dict = {}
    if height:
        graph_style["height"] = f"{height}px"
    return html.Div(
        style=style,
        children=[
            dcc.Graph(
                figure=figure,
                style=graph_style,
                config={"displayModeBar": False, "responsive": True},
            )
        ],
    )


def empty_state(message: str = "No data matches the selected filters."):
    return html.Div(
        message,
        style={
            "gridColumn": "1 / -1",
            "textAlign": "center",
            "padding": "60px 0",
            "color": TEXT_MUTED,
            "fontSize": "14px",
        },
    )


# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------

df_clean     = load_dashboard_data()
model_outputs = try_load_model_outputs()

all_states        = sorted(df_clean["customer_state"].dropna().unique())
all_years         = sorted(df_clean["year"].dropna().astype(int).unique())
all_payment_types = [
    p for p in ["Credit card", "Boleto", "Voucher", "Debit card"]
    if p in df_clean["payment_label"].unique()
]

app = dash.Dash(__name__, title="Olist Analytics")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "'Inter','Segoe UI',sans-serif", "background": BG_PAGE, "minHeight": "100vh"},
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": BG_CARD,
                "padding": "14px 28px",
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "alignItems": "center",
                "borderBottom": "1px solid #E8E8F0",
                "boxShadow": "0 1px 6px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div(
                    children=[
                        html.Span(
                            "Olist",
                            style={"fontSize": "18px", "fontWeight": "800", "color": BRAND_BLUE},
                        ),
                        html.Span(
                            " Analytics",
                            style={"fontSize": "18px", "fontWeight": "400", "color": TEXT_PRIMARY},
                        ),
                        html.Div(
                            "Customer Retention · Payment Behaviour · Delivery Risk",
                            style={"fontSize": "10px", "color": TEXT_MUTED, "marginTop": "2px"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                # Filters
                html.Div(
                    [
                        html.Div("States", style={"fontSize": "10px", "color": TEXT_MUTED, "marginBottom": "4px"}),
                        dcc.Dropdown(
                            id="f-state",
                            options=[{"label": s, "value": s} for s in all_states],
                            multi=True,
                            placeholder="All states",
                            style={"minWidth": "200px", "fontSize": "12px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Payment", style={"fontSize": "10px", "color": TEXT_MUTED, "marginBottom": "4px"}),
                        dcc.Checklist(
                            id="f-pay",
                            options=[{"label": f"  {p}", "value": p} for p in all_payment_types],
                            value=all_payment_types,
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "12px", "fontSize": "12px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Year", style={"fontSize": "10px", "color": TEXT_MUTED, "marginBottom": "4px"}),
                        dcc.Checklist(
                            id="f-year",
                            options=[{"label": f"  {y}", "value": y} for y in all_years],
                            value=all_years,
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "12px", "fontSize": "12px"},
                        ),
                    ]
                ),
            ],
        ),

        # ── KPI row ─────────────────────────────────────────────────────────
        html.Div(
            id="kpi-row",
            style={"display": "flex", "gap": "14px", "padding": "20px 28px 10px 28px", "flexWrap": "wrap"},
        ),

        # ── Tabs ────────────────────────────────────────────────────────────
        html.Div(
            style={"padding": "0 28px", "background": BG_CARD, "borderBottom": "1px solid #E8E8F0"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-overview",
                    style={"border": "none"},
                    children=[
                        dcc.Tab(label="Overview",           value="tab-overview",  style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                        dcc.Tab(label="Payment Behaviour",  value="tab-payment",   style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                        dcc.Tab(label="Delivery & Satisfaction", value="tab-delivery", style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                        dcc.Tab(label="Trends",             value="tab-trends",    style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                        dcc.Tab(label="Risk Overview",      value="tab-risk",      style=TAB_STYLE, selected_style=SELECTED_TAB_STYLE),
                    ],
                )
            ],
        ),

        # ── Tab content ─────────────────────────────────────────────────────
        html.Div(
            id="tab-content",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "padding": "20px 28px 32px 28px",
            },
        ),
    ],
)


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

@callback(
    Output("kpi-row", "children"),
    Output("tab-content", "children"),
    Input("f-state", "value"),
    Input("f-pay", "value"),
    Input("f-year", "value"),
    Input("tabs", "value"),
)
def update_dashboard(selected_states, payment_types, years, tab):
    data = df_clean.copy()

    if selected_states:
        data = data[data["customer_state"].isin(selected_states)]
    if payment_types:
        data = data[data["payment_label"].isin(payment_types)]
    if years:
        data = data[data["year"].isin(years)]

    if data.empty:
        return [], [empty_state()]

    # KPI row — accent colours communicate direction at a glance
    late_pct  = data["is_late"].mean() * 100
    avg_score = data["review_score"].mean()
    avg_delta = data["delivery_delta"].mean()
    kpis = [
        card("Orders",             f"{len(data):,}",                accent=BRAND_BLUE),
        card("Avg Review Score",   f"{avg_score:.2f} / 5",          accent=BRAND_GREEN if avg_score >= 4 else BRAND_ORANGE),
        card("Avg Delivery Delta", f"{avg_delta:.1f} d",            accent=BRAND_GREEN if avg_delta >= 0 else BRAND_RED),
        card("Late Orders",        f"{late_pct:.1f}%",              accent=BRAND_RED if late_pct > 10 else BRAND_ORANGE),
    ]

    layout = dict(CHART_LAYOUT)

    # ── Overview ────────────────────────────────────────────────────────────
    if tab == "tab-overview":
        cat_inst = (
            data.dropna(subset=["product_category"])
            .groupby("product_category")["payment_installments"]
            .median()
            .reset_index()
            .sort_values("payment_installments", ascending=False)
            .head(10)
        )
        fig_a = px.bar(
            cat_inst,
            x="payment_installments",
            y="product_category",
            orientation="h",
            title="Top 10 Categories by Median Installments",
            color_discrete_sequence=[BRAND_BLUE],
        )
        fig_a.update_layout({**layout, "height": 360, "yaxis": {"categoryorder": "total ascending"}})
        fig_a.update_traces(marker_line_width=0)

        pay_share = data.groupby(["customer_state", "payment_label"]).size().reset_index(name="count")
        pay_share["share"] = (
            pay_share["count"]
            / pay_share.groupby("customer_state")["count"].transform("sum")
            * 100
        )
        fig_b = px.bar(
            pay_share,
            x="customer_state",
            y="share",
            color="payment_label",
            color_discrete_map=COLOR_MAP,
            title="Payment Method Share by State (%)",
        )
        fig_b.update_layout({**layout, "height": 360, "barmode": "stack", "xaxis_tickangle": -45})

        review_counts = (
            data["review_score"]
            .value_counts()
            .reindex([1, 2, 3, 4, 5])
            .reset_index()
        )
        review_counts.columns = ["review_score", "count"]
        review_counts["pct"] = review_counts["count"] / review_counts["count"].sum() * 100
        review_counts["color"] = review_counts["review_score"].map({
            1: BRAND_RED, 2: BRAND_ORANGE, 3: "#F5C842", 4: "#74C69D", 5: BRAND_GREEN,
        })
        fig_reviews = go.Figure(go.Bar(
            x=review_counts["review_score"].astype(str),
            y=review_counts["count"],
            marker_color=review_counts["color"],
            text=[f"{p:.1f}%" for p in review_counts["pct"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_reviews.update_layout({
            **layout,
            "title": "Review Score Distribution",
            "xaxis_title": "Stars",
            "yaxis_title": "Orders",
            "height": 340,
            "showlegend": False,
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "margin": dict(l=12, r=12, t=44, b=12),
        })
        fig_reviews.update_traces(marker_line_width=0)

        return kpis, [
            chart_card(fig_reviews),
            chart_card(fig_a),
            chart_card(fig_b, span_full=True),
        ]

    # ── Payment Behaviour ───────────────────────────────────────────────────
    if tab == "tab-payment":
        fig_hist = px.histogram(
            data,
            x="payment_installments",
            nbins=24,
            title="Installment Distribution",
            color_discrete_sequence=[BRAND_BLUE],
        )
        fig_hist.update_layout({**layout, "height": 310})
        fig_hist.update_traces(marker_line_width=0.5, marker_line_color="white")

        corr_cols = [
            "payment_installments",
            "total_order_value",
            "total_freight",
            "delivery_delta",
            "delivery_speed_days",
            "is_late",
            "review_score",
            "customer_lifetime_orders",
            "review_sentiment",
        ]
        corr_cols = [c for c in corr_cols if c in data.columns]
        corr_matrix = data[corr_cols].corr().round(2)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        fig_corr = px.imshow(
            corr_matrix.where(~mask),
            text_auto=True,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix (Lower Triangle)",
        )
        fig_corr.update_layout({**layout, "height": 460})

        return kpis, [chart_card(fig_hist, span_full=True), chart_card(fig_corr, span_full=True)]

    # ── Delivery & Satisfaction ─────────────────────────────────────────────
    if tab == "tab-delivery":
        map_data = (
            df_clean
            .groupby("customer_state")
            .agg(total=("order_id", "count"), late=("is_late", "sum"))
            .reset_index()
        )
        map_data["late_rate"] = (map_data["late"] / map_data["total"]) * 100

        if selected_states:
            map_data["display"] = map_data.apply(
                lambda r: r["late_rate"] if r["customer_state"] in selected_states else None,
                axis=1,
            )
        else:
            map_data["display"] = map_data["late_rate"]

        fig_map = px.choropleth(
            map_data,
            geojson=BRAZIL_GEOJSON,
            locations="customer_state",
            featureidkey="properties.sigla",
            color="display",
            color_continuous_scale="Reds",
            title="Late Delivery % by State",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout({**layout, "height": 460})

        fig_violin = go.Figure()
        for label, color in COLOR_MAP.items():
            subset = data[data["payment_label"] == label]
            if not subset.empty:
                fig_violin.add_trace(
                    go.Violin(
                        x=subset["payment_label"],
                        y=subset["review_score"],
                        name=label,
                        line_color=color,
                        fillcolor=color,
                        opacity=0.7,
                        points=False,
                        box_visible=True,
                    )
                )
        fig_violin.update_layout({
            **layout,
            "title": "Review Score by Payment Type",
            "height": 340,
            "showlegend": False,
        })

        fig_box = px.box(
            data,
            x="payment_label",
            y="delivery_delta",
            color="payment_label",
            color_discrete_map=COLOR_MAP,
            title="Delivery Delta vs Estimate (days)",
        )
        fig_box.update_layout({**layout, "height": 340, "showlegend": False})

        late_state = (
            data.groupby("customer_state")["is_late"]
            .mean()
            .reset_index()
            .sort_values("is_late", ascending=False)
        )
        late_state["late_pct"] = late_state["is_late"] * 100
        fig_bar = px.bar(
            late_state,
            x="customer_state",
            y="late_pct",
            title="Late Rate by State (filtered selection)",
            color="late_pct",
            color_continuous_scale="Reds",
            labels={"late_pct": "Late %"},
        )
        fig_bar.update_layout({**layout, "height": 340, "coloraxis_showscale": False})
        fig_bar.update_traces(marker_line_width=0)

        return kpis, [
            chart_card(fig_map, span_full=True),
            chart_card(fig_violin),
            chart_card(fig_box),
            chart_card(fig_bar, span_full=True),
        ]

    # ── Trends ──────────────────────────────────────────────────────────────
    if tab == "tab-trends":
        trend = (
            data.groupby(["year_month", "payment_label"])["review_score"]
            .mean()
            .reset_index()
        )
        fig_trend = px.line(
            trend,
            x="year_month",
            y="review_score",
            color="payment_label",
            color_discrete_map=COLOR_MAP,
            title="Average Review Score Over Time by Payment Method",
            markers=True,
        )
        fig_trend.update_layout({
            **layout,
            "height": 420,
            "xaxis_tickangle": -45,
            "hovermode": "x unified",
        })
        fig_trend.update_traces(line_width=2)

        # Avg order value trend
        val_trend = (
            data.groupby(["year_month", "payment_label"])["total_order_value"]
            .mean()
            .reset_index()
        )
        fig_val = px.line(
            val_trend,
            x="year_month",
            y="total_order_value",
            color="payment_label",
            color_discrete_map=COLOR_MAP,
            title="Average Order Value Over Time by Payment Method",
            markers=True,
        )
        fig_val.update_layout({
            **layout,
            "height": 420,
            "xaxis_tickangle": -45,
            "hovermode": "x unified",
        })
        fig_val.update_traces(line_width=2)

        return kpis, [chart_card(fig_trend, span_full=True), chart_card(fig_val, span_full=True)]

    # ── Risk Overview ────────────────────────────────────────────────────────
    if tab == "tab-risk":
        if not model_outputs:
            return kpis, [
                empty_state(
                    "Run the ML workflow in the notebook first to generate outputs/model/ artifacts."
                )
            ]

        children: list = []

        # Model metrics table
        if "metrics" in model_outputs:
            metrics_df = model_outputs["metrics"]
            display_cols = [c for c in [
                "model", "decision_threshold", "f1_low_satisfaction",
                "recall_low_satisfaction", "precision_low_satisfaction", "roc_auc",
            ] if c in metrics_df.columns]
            fig_metrics = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=[c.replace("_", " ").title() for c in display_cols],
                            fill_color=BRAND_BLUE,
                            font=dict(color="white", size=12),
                            align="left",
                            height=30,
                        ),
                        cells=dict(
                            values=[metrics_df[c].round(4) if metrics_df[c].dtype != object else metrics_df[c] for c in display_cols],
                            fill_color=[[BG_CARD, "#F5F7FA"] * (len(metrics_df) + 1)],
                            align="left",
                            height=28,
                            font=dict(color=TEXT_PRIMARY, size=11),
                        ),
                    )
                ]
            )
            fig_metrics.update_layout({**layout, "title": "Model Performance (Test Set)", "height": 220})
            children.append(chart_card(fig_metrics, span_full=True))

        # Risk distribution
        if "risk_table" in model_outputs:
            rt = model_outputs["risk_table"]
            if "predicted_risk_band" in rt.columns:
                risk_counts = rt["predicted_risk_band"].value_counts().reset_index()
                risk_counts.columns = ["risk_band", "count"]
                color_seq = {
                    "high": BRAND_RED, "medium": BRAND_ORANGE, "low": BRAND_GREEN,
                }
                fig_pie = px.pie(
                    risk_counts,
                    names="risk_band",
                    values="count",
                    title="Risk Band Distribution",
                    color="risk_band",
                    color_discrete_map=color_seq,
                    hole=0.42,
                )
                fig_pie.update_layout({**layout, "height": 340})
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")

                # Top high-risk states
                if "customer_state" in rt.columns:
                    high_risk_state = (
                        rt[rt["predicted_risk_band"] == "high"]
                        .groupby("customer_state")
                        .size()
                        .reset_index(name="high_risk_orders")
                        .sort_values("high_risk_orders", ascending=False)
                        .head(15)
                    )
                    fig_risk_state = px.bar(
                        high_risk_state,
                        x="customer_state",
                        y="high_risk_orders",
                        title="High-Risk Orders by State",
                        color_discrete_sequence=[BRAND_RED],
                    )
                    fig_risk_state.update_layout({**layout, "height": 340})
                    fig_risk_state.update_traces(marker_line_width=0)
                    children += [chart_card(fig_pie), chart_card(fig_risk_state)]
                else:
                    children.append(chart_card(fig_pie))

        # Segment profile heatmap
        if "segment_profile" in model_outputs:
            seg = model_outputs["segment_profile"]
            heatmap_cols = [c for c in [
                "payment_installments", "total_order_value", "delivery_delta",
                "delivery_speed_days", "is_late", "customer_lifetime_orders",
                "high_satisfaction", "average_review_score",
            ] if c in seg.columns]
            label_col = "segment_label" if "segment_label" in seg.columns else "segment"
            if heatmap_cols:
                fig_seg = px.imshow(
                    seg.set_index(label_col)[heatmap_cols].round(2),
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    title="Customer Segment Profile Heatmap",
                )
                fig_seg.update_layout({**layout, "height": 300})
                children.append(chart_card(fig_seg, span_full=True))

        return kpis, children if children else [empty_state("No model artifacts found.")]

    # Fallback — should never reach here
    return kpis, [empty_state("Unknown tab.")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
