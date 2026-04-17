# Improving Customer Retention for Olist Ecommerce Platform
### A Data Science & GenAI Project on the Brazilian E-Commerce Market

---

## Overview

This project investigates whether **payment behaviour** (payment method and number of installments) influences **customer satisfaction** and **delivery experience** in the Olist Brazilian e-commerce ecosystem.

The analysis spans the full data science pipeline: from exploratory analysis and feature engineering, to machine learning classification, an interactive BI dashboard, and a GenAI-powered automated customer recovery agent.

> **Business question:** Can we predict which customers are at risk of churning based on how they pay — and automatically intervene with a personalised recovery message?

---

## Presentation

▶ **[Watch the project presentation](https://drive.google.com/file/d/1QIId3io7uQoxckcEW1gD8x6gDsWR1AJW/view?usp=sharing)**

The full walkthrough covers the business problem, EDA findings, ML results, and live demo of the customer recovery agent. The video file (`Improving_Customer_Retention.mp4`) is also included in the repository root.

---

## Live App

The customer recovery agent is deployed on Streamlit Community Cloud:

**[emailolist.streamlit.app](https://emailolist.streamlit.app)**

> Requires an `OPENAI_API_KEY` added via the Streamlit Secrets dashboard to generate emails.

---

## Project Structure

```
Improving_Customer_Retention_through_Payment_Analytics/
│
├── .devcontainer/
│   └── devcontainer.json                    # GitHub Codespaces config — auto-installs deps & launches app.py
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
│       ├── segment_profile.csv
│       ├── predictions.csv  (gitignored)
│       └── risk_table.csv   (gitignored — regenerate by running the notebook)
│
├── app.py                                   # Streamlit Customer Recovery Agent
├── dashboard.py                             # Interactive Dash BI dashboard
├── Improving_Customer_Retention.mp4         # Project presentation video
├── requirements.txt                         # Python dependencies for deployment
└── README.md
```

> `environment.yml` is kept locally for conda development but is not tracked in the repo. Use `requirements.txt` for all deployments.

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

Tables are joined on `order_id` and `customer_id` to produce a single flat analytical dataframe. Payment records are aggregated per order (dominant payment type by value, total installments, total payment value). Reviews are deduplicated — most recent kept per order. Only `delivered` orders are used.

### 2. Feature Engineering

| Feature | Description |
|---|---|
| `delivery_delta` | Estimated delivery date minus actual delivery date (positive = early, negative = late) |
| `delivery_speed_days` | Days from purchase to actual delivery |
| `is_late` | Binary flag: actual delivery exceeded estimated date |
| `total_order_value` | Sum of item prices + freight for the order |
| `customer_lifetime_orders` | Number of historical orders per `customer_unique_id` |
| `payment_type_*` | One-hot encoded payment method columns |
| `review_sentiment` | TextBlob polarity score from `review_comment_message` (0 if no comment) |

### 3. Exploratory Data Analysis

All EDA uses **Plotly** for interactive visualisations (available both in the notebook and the live dashboard):

- **Review score distribution** — bar chart showing count and percentage per star rating (1–5), colour-coded from red to green
- **Installment distribution by product category**
- **Payment method share by state** (choropleth)
- **Correlation with review_score** — standalone horizontal bar chart (Pearson r) for every numeric feature against `review_score`, colour-coded red/blue by direction; saved to `outputs/figures/correlation_review_score.png`
- **Correlation matrix** — full lower-triangle heatmap of all numeric features
- **Review score by payment type** — violin plot
- **Satisfaction trend over time** — average review score by month split by payment type

**Key narrative finding:** EDA and SHAP analysis both confirmed that delivery experience (`delivery_delta`, `is_late`, `delivery_speed_days`) is a far stronger predictor of satisfaction than payment behaviour. This challenged the original hypothesis and redirects retention strategy toward logistics quality over payment incentives.

### 4. Machine Learning

**Target variable:** Binary satisfaction label — `high` (4–5 stars) vs `low` (1–3 stars)

**Feature scope options:**

| Scope | Features | Use case |
|---|---|---|
| `post_delivery` | All features incl. `delivery_delta`, `delivery_speed_days`, `is_late` | Post-purchase recovery targeting |
| `pre_delivery` | Payment and order-value features only | Earlier risk prediction, no leakage |

**Models trained:**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline, `class_weight="balanced"` |
| Random Forest | Ensemble benchmark, `class_weight="balanced_subsample"` |
| XGBoost | Primary model — regularised, with `scale_pos_weight` |

**XGBoost configuration:** `max_depth=4`, `min_child_weight=10`, `gamma=0.1`, `reg_alpha=0.1`, `reg_lambda=2.0`. `scale_pos_weight` is computed automatically from training class distribution.

**Evaluation:** AUC-ROC plus threshold-tuned F1. Decision threshold tuned on a held-out validation split (sweep 0.20–0.81) to maximise low-satisfaction F1, then applied on the test set.

**Cross-validation:** `compute_cv_metrics()` runs stratified 5-fold CV for all three models and reports mean ± std for AUC, F1, recall, and precision.

**Interpretability:** SHAP values via `compute_shap_artifacts()` show directional feature impact per prediction.

**Customer segmentation:** K-Means clustering on payment and delivery behaviour. Optimal K selected via silhouette score sweep (K=2–8).

**Cost-sensitive variant:** `train_vip_weighted_xgboost()` applies `sample_weight` proportional to order value to reduce expensive false negatives on high-value customers.

**Operational outputs:**

- `build_revenue_at_risk_table()` + `build_revenue_at_risk_map()` — state-level revenue concentration for high-value late deliveries.
- `prepare_intervention_payload()` — enriched agent input with personalisation fields (product category, delay metrics, value tier, risk %, recommended coupon %).

### 5. Interactive Dashboard

Run the Dash BI dashboard locally:

```bash
python dashboard.py
# Opens at http://localhost:8050
```

Or launch it directly from the notebook by running the **"Launch the EDA Dashboard"** cell (inserted after the EDA figures cell). This starts `dashboard.py` as a background subprocess so the kernel stays interactive. Stop it with `_dashboard_proc.terminate()`.

**Tabs:**

| Tab | Contents |
|---|---|
| Overview | Review score distribution; top categories by median installments; payment method share by state |
| Payment Behaviour | Installment distribution; full correlation matrix |
| Delivery & Satisfaction | Late-delivery choropleth; review score violin; delivery delta box; late-rate bar |
| Trends | Avg review score and avg order value over time by payment method |
| Risk Overview | Model performance table; risk-band distribution; high-risk orders by state; segment heatmap |

All tabs respond to the **State**, **Payment**, and **Year** filter controls in the header.

### 6. GenAI Customer Recovery Agent

For orders classified as **high churn risk**, a personalised recovery email is generated using the **OpenAI API** (`gpt-4o-mini`). The full implementation lives in two files:

**`src/genai.py`** — core module (OpenAI SDK, no third-party LLM frameworks):

| Function | Description |
|---|---|
| `generate_recovery_email(customer_row, tone)` | Sends a structured prompt to `gpt-4o-mini` and returns a personalised email body (≤ 150 words). The prompt includes the customer's state, product category, payment method and installment count, days late, order value in BRL, and the recommended discount. The `tone` parameter (`"empathetic"` / `"formal"` / `"friendly"`) adjusts the writing style. |
| `build_customer_summary(customer_row)` | Returns a 4-bullet plain-text summary of the order for display in the app. |

**`app.py`** — Streamlit front-end:

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The app loads `notebooks/outputs/intervention/intervention_payload_high_risk_late.csv` and provides:

- **Sidebar filters:** risk band, customer state (multiselect), and email tone (Empathetic / Formal / Friendly)
- **Main table:** filtered customers with order ID, state, product category, order value (R$), days late, coupon %, and risk band
- **Email generator:** select a customer, click Generate or Regenerate; results are cached in `st.session_state` so widget interactions don't re-trigger the API call

**Example output:**
> *Dear Customer, we sincerely apologise that your furniture living room order — paid by credit card in a single payment (R$ 255.96) — arrived 29 days late. This falls well short of the experience you deserve. As a thank-you for your patience, we're offering you 25% off your next purchase. — Olist Customer Success Team*

---

## Setup

### Option A — pip (recommended)

```bash
pip install -r requirements.txt
jupyter notebook
```

### Option B — Conda (local development)

An `environment.yml` is included locally but not tracked in the repo. To recreate the conda environment:

```bash
conda env create -f environment.yml
conda activate project-env
jupyter notebook
```

### Kaggle credentials

The notebook downloads the dataset automatically via `kagglehub`. Place your API credentials at `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

Credentials can be downloaded from your [Kaggle account settings](https://www.kaggle.com/settings/account).

### OpenAI API key

Required for the GenAI recovery agent. Set it as an environment variable before running the app:

**Mac / Linux:**
```bash
export OPENAI_API_KEY="your_key_here"
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_key_here"
streamlit run app.py
```

**Streamlit Community Cloud:** Add the key in the app's **Settings → Secrets** tab:
```toml
OPENAI_API_KEY = "your_key_here"
```

---

## Key Findings

- **Best classifier:** XGBoost (`f1_low_satisfaction=0.435`, `recall_low_satisfaction=0.492`, `roc_auc=0.693`)
- **The data challenged the hypothesis:** Delivery experience (`delivery_delta`, `is_late`, `delivery_speed_days`) is the primary driver of satisfaction — not payment behaviour. Payment features contribute marginal signal.
- **Risk concentration:** 89,958 low-risk · 1,507 medium-risk · 4,358 high-risk orders
- **Top retention target:** Segment 1 (`late-delivery risk`) — 65.6% low-satisfaction rate across 7,604 orders, avg review 2.56 stars
- **VIP-weighted model:** recall improved 0.461 → 0.485; false-negative revenue reduced R$126k → R$120k
- **Revenue at risk:** Largest concentrations in SP (São Paulo) and RJ (Rio de Janeiro)
- **Intervention export:** 4,358 high-risk orders with full personalisation metadata (delay days, value tier, risk %, coupon %)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| NLP | `textblob` |
| Visualisation | `plotly`, `matplotlib` |
| Machine learning | `scikit-learn`, `xgboost`, `shap` |
| Dashboarding | `dash` |
| GenAI | `openai` (GPT-4o-mini) |
| App deployment | `streamlit`, Streamlit Community Cloud |
| Environment | `conda`, `jupyter`, Python 3.11 |

---

## Author

**Laura Romero**
[LinkedIn](https://www.linkedin.com/in/laura-romero-gonzalez) · [GitHub](https://github.com/lauraromerog)

---

## License

This project is for educational and portfolio purposes. The dataset is provided by Olist and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
