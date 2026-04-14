# Improving Customer Retention through Payment Analytics
### A Data Science & GenAI Project on the Brazilian E-Commerce Market

---

## Overview

This project investigates whether **payment behaviour** (payment method and number of installments) influences **customer satisfaction** and **delivery experience** in the Olist Brazilian e-commerce ecosystem.

The analysis spans the full data science pipeline: from exploratory analysis and feature engineering, to machine learning classification, an interactive BI dashboard, and a GenAI-powered automated customer recovery agent.

> **Business question:** Can we predict which customers are at risk of churning based on how they pay — and automatically intervene with a personalised recovery message?

---

## Project Structure

```
Improving_Customer_Retention_through_Payment_Analytics/
│
├── notebooks/
│   ├── olist_payment_analytics.ipynb        # Main analysis notebook
│   └── outputs/intervention/
│       └── intervention_payload_high_risk_late.csv
│
├── src/
│   ├── __init__.py                          # Public API exports
│   ├── data_loader.py                       # Dataset loading & merging utilities
│   ├── feature_engineering.py               # Feature creation (delivery_delta, etc.)
│   ├── model.py                             # ML training, evaluation & segmentation
│   └── genai.py                             # OpenAI email generation & customer summary
│
├── outputs/
│   └── model/                               # Saved artifacts (metrics, predictions, models)
│       ├── metrics.csv
│       ├── predictions.csv
│       ├── risk_table.csv
│       └── segment_profile.csv
│
├── app.py                                   # Streamlit Customer Recovery Agent
├── dashboard.py                             # Interactive Dash BI dashboard
├── requirements.txt
├── README.md
└── environment.yml
```

---

## Dataset

**Source:** [Olist Brazilian E-Commerce — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

~100k real anonymised orders placed on the Olist marketplace between 2016 and 2018. Seven tables are joined to build the analytical dataset:

| Table | Purpose |
|---|---|
| `olist_orders_dataset` | Order status, purchase and delivery timestamps |
| `olist_order_payments_dataset` | Payment type and number of installments |
| `olist_order_reviews_dataset` | Customer review scores (1–5 stars) |
| `olist_customers_dataset` | Customer city and state |
| `olist_order_items_dataset` | Item price, freight value, and product category |
| `olist_products_dataset` | Product metadata |
| `product_category_name_translation` | English category name translations |

---

## Methodology

### 1. Data Strategy

Tables are joined on `order_id` and `customer_id` to produce a single flat analytical dataframe. Payment records are aggregated per order (dominant payment type, total installments, total payment value).

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `delivery_delta` | Estimated delivery date minus actual delivery date (positive = delivered early) |
| `delivery_speed_days` | Days from purchase to actual delivery |
| `is_late` | Binary flag: actual delivery exceeded estimated date |
| `total_order_value` | Sum of item prices + freight for the order |
| `customer_lifetime_orders` | Number of historical orders per customer |
| `payment_type_*` | One-hot encoded payment method columns |

### 3. Exploratory Data Analysis

All EDA uses **Plotly** for interactive visualisations (available both in the notebook and the live dashboard):

- **Installment distribution by product category**
- **Payment method share by state** (choropleth)
- **Correlation matrix** — numeric features including installments, order value, delivery delta, and review score (`delivery_speed_days` now included)
- **Review score by payment type** — violin plot
- **Satisfaction trend over time** — average review score by month split by payment type

### 4. Machine Learning

**Target variable:** Binary satisfaction label — `high` (4–5 stars) vs `low` (1–3 stars)

**Feature scope options:**

| Scope | Features | Use case |
|---|---|---|
| `post_delivery` | All features incl. `delivery_delta`, `delivery_speed_days`, `is_late` | Post-purchase recovery targeting |
| `pre_delivery` | Payment and order-value features only | Earlier risk prediction, no delivery leakage |

**Models trained:**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline, `class_weight="balanced"` |
| Random Forest | Ensemble benchmark, `class_weight="balanced_subsample"` |
| XGBoost | Primary model — regularised, with `scale_pos_weight` |

**XGBoost improvements (latest version):**

The XGBoost configuration was upgraded to reduce overfitting on noisy order-level data and to handle class imbalance natively, consistent with how sklearn estimators use `class_weight="balanced"`:

- `max_depth` reduced from 5 → 4
- Added `min_child_weight=10` (prevents splits on very small leaf nodes)
- Added `gamma=0.1` (minimum gain required to make a split)
- Added `reg_alpha=0.1`, `reg_lambda=2.0` (L1/L2 regularisation)
- `scale_pos_weight` is now computed automatically from the training class distribution (`n_low_satisfaction / n_high_satisfaction`)
- `delivery_speed_days` added to `POST_DELIVERY_NUMERIC_FEATURES` (previously missing)

**Evaluation:** AUC-ROC plus threshold-tuned F1, with model ranking driven by **low-satisfaction F1** and recall rather than raw accuracy. The decision threshold is tuned on a held-out validation split to improve detection of at-risk customers.

**Cross-validation:** `compute_cv_metrics()` runs stratified 5-fold CV for all three models and reports mean ± std for AUC, low-satisfaction F1, recall, and precision. Use this alongside `train_satisfaction_models()` to confirm that hold-out results generalise across folds.

**Interpretability:** SHAP values (via `compute_shap_artifacts()`) show the directional impact of each feature on predicted satisfaction.

**Customer segmentation:** K-Means clustering on payment and delivery behaviour identifies distinct risk profiles (e.g., high-value reliable payers, installment-heavy late-delivery risk). Optimal K is selected using a silhouette score sweep (`score_cluster_counts()`).

**Risk output:** A ranked customer/order risk table with predicted low-satisfaction probability and risk bands (`low`, `medium`, `high`) for downstream retention actions.

**Cost-sensitive variant:** `train_vip_weighted_xgboost()` applies `sample_weight` to prioritise high-value segments and reduce expensive false negatives.

**Operational outputs:**

- `build_revenue_at_risk_table()` + `build_revenue_at_risk_map()` — state-level revenue concentration for high-value late deliveries.
- `prepare_intervention_payload()` — enriched email-agent input with personalisation fields (product category, delay metrics, value tier, risk %, recommended coupon %).

### 5. Interactive Dashboard

Run the Dash BI dashboard locally:

```bash
python dashboard.py
# Opens at http://localhost:8050
```

**Tabs:**

| Tab | Contents |
|---|---|
| Overview | Top categories by median installments; payment method share by state |
| Payment Behaviour | Installment distribution; full correlation matrix (incl. `delivery_speed_days`) |
| Delivery & Satisfaction | Late-delivery choropleth; review score violin; delivery delta box; late-rate bar |
| Trends | Avg review score and avg order value over time by payment method |
| Risk Overview | Model performance table; risk-band distribution; high-risk orders by state; segment heatmap — loaded automatically from `outputs/model/` if the ML workflow has been run |

All tabs respond to the **State**, **Payment**, and **Year** filter controls in the header.

### 6. GenAI Customer Recovery Agent

For orders classified as **high churn risk**, a personalised recovery email is generated using the **OpenAI API** (`gpt-4o-mini`). The full implementation lives in two files:

**`src/genai.py`** — core module (OpenAI SDK only, no third-party LLM frameworks):

| Function | Description |
|---|---|
| `generate_recovery_email(customer_row, tone)` | Sends a structured prompt to `gpt-4o-mini` and returns the email body (≤ 150 words). The prompt includes the customer's state, product category, payment method and installment count, days late, order value in BRL, and the recommended discount. The `tone` parameter (`"empathetic"` / `"formal"` / `"friendly"`) adjusts the writing style via the system prompt. |
| `build_customer_summary(customer_row)` | Returns a 4-bullet plain-text summary of the order (value, days late, churn risk %, product, payment method) for display in the app sidebar. |

**`app.py`** — Streamlit front-end:

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The app loads `notebooks/outputs/intervention/intervention_payload_high_risk_late.csv` and provides:

- **Sidebar filters:** risk band (high / medium / low), customer state (multiselect), and email tone radio button; a metric shows the number of matching customers.
- **Main table:** filtered customers showing order ID, state, product category, order value (R$), days late, coupon %, and risk band.
- **Email generator:** select a customer from a dropdown, click **Generate Email ✉️** or **🔄 Regenerate**; the result is displayed in a tall text area with an order-summary info box above it. Results are stored in `st.session_state` so widget interactions don't re-trigger the API call.

> **Requires** `OPENAI_API_KEY` set as an environment variable before launching:
> ```bash
> export OPENAI_API_KEY="your_key_here"
> streamlit run app.py
> ```

**Example output:**
> *Dear Customer, we sincerely apologise that your furniture living room order — paid by credit card in a single payment (R$ 255.96) — arrived 29 days late. This falls well short of the experience you deserve. As a thank-you for your patience, we're offering you 25% off your next purchase. Use code at checkout — we look forward to making it right. — Olist Customer Success Team*

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate project-env
jupyter notebook
```

### Option B — pip

```bash
pip install pandas numpy scikit-learn xgboost shap plotly dash kagglehub textblob openai
jupyter notebook
```

### Kaggle credentials

The notebook downloads the dataset automatically via `kagglehub`. Place your API credentials at `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

Credentials can be downloaded from your [Kaggle account settings](https://www.kaggle.com/settings/account).

### API key (for GenAI component)

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## Key Findings

Latest notebook run (full `src` workflow):

- **Best classifier:** XGBoost ranked first for low-satisfaction detection (`f1_low_satisfaction=0.422`, `recall_low_satisfaction=0.461`, `roc_auc=0.678`) with a tuned decision threshold of `0.79`. Updated regularisation and `scale_pos_weight` calibration are expected to improve these figures on the next full run.
- **Risk concentration:** `89,958` low-risk, `1,507` medium-risk, and `4,358` high-risk orders.
- **Top retention target segment:** Segment `1` (`late-delivery risk | satisfaction risk`) — ~65.6% low-satisfaction rate across `7,604` orders.
- **Most influential model drivers (SHAP):** `delivery_delta`, `total_freight`, and `is_late` are the strongest predictors. Delivery performance has a larger impact than payment dummies alone.
- **Payment behaviour signal:** Payment features (installments, payment-method dummies) contribute to prediction but with smaller effect size than delivery variables.
- **Key narrative finding:** The data revealed that delivery experience is a stronger churn predictor than payment behaviour — `delivery_delta`, `is_late`, and `delivery_speed_days` consistently outrank payment features in SHAP importance. This is a discovery about where churn risk actually originates, not a limitation: it redirects retention efforts from payment incentives toward logistics and fulfilment quality.
- **VIP-weighted training improved business risk handling:** low-satisfaction recall increased `0.461 → 0.485`; VIP false-negative revenue decreased `$126,370 → $120,305`.
- **Revenue-at-risk map:** Largest concentrations in `SP` (São Paulo) and `RJ` (Rio de Janeiro).
- **Intervention export:** `4,358` high-risk delayed-shipping orders exported to `outputs/intervention/intervention_payload_high_risk_late.csv` with full personalisation metadata.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `plotly`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost`, `shap` |
| Dashboarding | `dash` |
| GenAI | `openai` |
| Environment | `conda`, `jupyter`, Python 3.11 |

---

## Author

**Laura Romero**  
[LinkedIn](https://www.linkedin.com/in/laura-romero-gonzalez) · [GitHub](https://github.com/lauraromerog)

---

## License

This project is for educational and portfolio purposes. The dataset is provided by Olist and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
