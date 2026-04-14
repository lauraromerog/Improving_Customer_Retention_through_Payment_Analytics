"""Streamlit Customer Recovery Agent — Olist high-risk order email generator.

Run:
    streamlit run app.py

Requires:
    OPENAI_API_KEY environment variable set before launching.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Recovery Agent",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYLOAD_PATH = (
    Path(__file__).parent
    / "notebooks"
    / "outputs"
    / "intervention"
    / "intervention_payload_high_risk_late.csv"
)

DISPLAY_COLUMNS = [
    "order_id",
    "customer_state",
    "product_category",
    "total_order_value",
    "delay_days",
    "recommended_coupon_pct",
    "predicted_risk_band",
]

TONE_OPTIONS = ["Empathetic", "Formal", "Friendly"]
RISK_BAND_OPTIONS = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_payload() -> pd.DataFrame:
    """Load the intervention payload CSV. Returns empty DataFrame if missing."""
    if not PAYLOAD_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PAYLOAD_PATH)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ Customer Recovery Agent")
st.caption(
    "Generate personalised recovery emails for high-risk Olist customers using GPT-4o-mini. "
    "Filter by risk band and state, select a customer, and let the agent do the writing."
)

# ---------------------------------------------------------------------------
# API key guard — check before any further work
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "Export your key in the terminal before launching this app:\n"
        "```bash\nexport OPENAI_API_KEY='your_key_here'\n```\n"
        "Then restart with `streamlit run app.py`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df_all = load_payload()

if df_all.empty:
    st.warning(
        "Intervention payload not found at "
        f"`{PAYLOAD_PATH.relative_to(Path(__file__).parent)}`.\n\n"
        "Run the ML workflow in `notebooks/olist_payment_analytics.ipynb` first "
        "to generate the file."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filters and tone
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    available_risk_bands = [
        b for b in RISK_BAND_OPTIONS
        if b in df_all["predicted_risk_band"].values
    ]
    risk_band = st.selectbox(
        "Risk band",
        options=available_risk_bands,
        index=0,
    )

    all_states = sorted(df_all["customer_state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "Customer state",
        options=all_states,
        default=all_states,
    )

    st.divider()

    st.header("Email tone")
    tone_label = st.radio(
        "Tone",
        options=TONE_OPTIONS,
        index=0,
        help=(
            "**Empathetic** — warm and apologetic\n\n"
            "**Formal** — professional and brief\n\n"
            "**Friendly** — casual and upbeat"
        ),
    )
    tone = tone_label.lower()

    st.divider()

    # Apply filters
    df_filtered: pd.DataFrame = df_all[df_all["predicted_risk_band"] == risk_band]
    if selected_states:
        df_filtered = df_filtered[df_filtered["customer_state"].isin(selected_states)]

    st.metric("Matching customers", len(df_filtered))

# ---------------------------------------------------------------------------
# Main area — customer table
# ---------------------------------------------------------------------------

if df_filtered.empty:
    st.info("No customers match the current filters. Adjust the sidebar selections.")
    st.stop()

# Build a display-friendly copy of the table
display_df = df_filtered[DISPLAY_COLUMNS].copy()
display_df["total_order_value"] = display_df["total_order_value"].map(
    lambda v: f"R$ {float(v):,.2f}"
)

st.subheader(f"Customers — {risk_band.capitalize()} Risk")
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Email generation
# ---------------------------------------------------------------------------

st.subheader("Generate Recovery Email")

order_ids = df_filtered["order_id"].tolist()
selected_order_id = st.selectbox(
    "Select a customer to generate their recovery email",
    options=order_ids,
)

col_gen, col_regen, _ = st.columns([1, 1, 5])
generate_btn = col_gen.button("Generate Email ✉️", type="primary")
regenerate_btn = col_regen.button("🔄 Regenerate")

# Initialise session state keys
for key, default in [
    ("email_text", ""),
    ("customer_summary", ""),
    ("email_order_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Trigger generation on either button
if generate_btn or regenerate_btn:
    customer_row = (
        df_filtered[df_filtered["order_id"] == selected_order_id]
        .iloc[0]
        .to_dict()
    )

    from src.genai import build_customer_summary, generate_recovery_email

    with st.spinner("Generating personalised email via GPT-4o-mini…"):
        try:
            email_text = generate_recovery_email(customer_row, tone=tone)
            summary_text = build_customer_summary(customer_row)
            st.session_state.email_text = email_text
            st.session_state.customer_summary = summary_text
            st.session_state.email_order_id = selected_order_id
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error while calling the OpenAI API: {exc}")

# Display results stored in session state (persists across widget interactions)
if st.session_state.email_text:
    if st.session_state.email_order_id:
        st.caption(f"Order ID: `{st.session_state.email_order_id}`")

    if st.session_state.customer_summary:
        st.info(st.session_state.customer_summary)

    st.text_area(
        "Generated email",
        value=st.session_state.email_text,
        height=290,
        key="email_display",
    )
