# Improving Customer Retention through Payment Analytics
### A Data Science & GenAI Project on the Brazilian E-Commerce Market

---

## Overview

This project investigates whether **payment behavior** (specifically payment method and number of installments) influences **customer satisfaction** and **delivery experience** in the Olist Brazilian e-commerce ecosystem.

The analysis spans the full data science pipeline: from exploratory analysis and feature engineering, to machine learning classification and a GenAI-powered automated customer recovery agent.

> **Business question:** Can we predict which customers are at risk of churning based on how they pay — and automatically intervene with a personalized recovery message?

---

## Project Structure

```
olist-payment-analytics/
│
├── data/                          # Downloaded automatically via kagglehub (not tracked in git)
│
├── notebooks/
│   └── olist_payment_analytics.ipynb   # Main analysis notebook
│
├── src/
│   ├── __init__.py               # Public exports for reusable project utilities
│   ├── data_loader.py             # Dataset loading & merging utilities
│   ├── feature_engineering.py     # Feature creation (delivery_delta, etc.)
│   ├── model.py                   # ML training & evaluation
│   └── email_agent.py             # GenAI customer recovery agent
│
├── outputs/
│   ├── figures/                   # Saved EDA charts
│   └── model/                     # Saved model artifacts
│
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Dataset

**Source:** [Olist Brazilian E-Commerce — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset contains ~100k real anonymized orders placed on the Olist marketplace between 2016 and 2018. Seven tables are joined to build the analytical dataset:

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

Tables are joined on `order_id` and `customer_id` to build a single flat analytical dataframe. Payment records are aggregated per order (dominant payment type, total installments, total payment value).

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

All EDA is done in Python using **Plotly** for interactive visualizations:

- **Installment distribution by product category** — bar chart of most common installment counts per category
- **Brazil payment method map** — choropleth by state showing dominant payment type (credit card / boleto / voucher)
- **Correlation matrix** — numeric features including installments, order value, delivery delta, and review score
- **Review score by payment type** — violin/box plot as direct evidence for the core hypothesis
- **Satisfaction trend over time** — average review score by month, split by payment type (2016–2018)

### 4. Machine Learning

**Target variable:** Binary satisfaction label — `high` (4–5 stars) vs `low` (1–3 stars)

**Input features:** `payment_type` (one-hot), `payment_installments`, `total_price`, `total_freight`, `delivery_delta`, `is_late`, `total_order_value`, `customer_lifetime_orders`

Two feature scopes are supported in code:

- `post_delivery` for post-purchase recovery targeting, where delivery outcome features such as `delivery_delta` and `is_late` are allowed
- `pre_delivery` for earlier prediction without delivery leakage

**Models trained:**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble benchmark |
| XGBoost | Primary model |

**Evaluation:** AUC-ROC plus threshold-tuned F1, with model ranking driven by **low-satisfaction F1** and recall rather than raw accuracy. The decision threshold is tuned on a validation split to improve detection of at-risk customers.

**Interpretability:** SHAP values are used to show the directional impact of each feature on predicted satisfaction, going beyond a simple feature importance bar chart.

**Customer segmentation:** K-Means clustering on payment and delivery behavior to identify distinct customer risk profiles (e.g., high-value reliable payers, installment-heavy late-delivery risk).

**Risk output:** The workflow also produces a ranked customer/order risk table with predicted low-satisfaction probability and risk bands (`low`, `medium`, `high`) for downstream retention actions.

**Cost-sensitive variant:** The project now includes a reusable VIP-weighted XGBoost training utility (`train_vip_weighted_xgboost`) that applies `sample_weight` to prioritize high-value segments and reduce expensive false negatives.

**Operational outputs:**

- `build_revenue_at_risk_table` + `build_revenue_at_risk_map` to prioritize states where high-value late deliveries concentrate.
- `prepare_intervention_payload` to export enriched email-agent input with personalization fields (including product category, delay metrics, value tier, risk %, and recommended coupon).

### 5. GenAI Customer Recovery Agent

For orders classified as **high churn risk** by the ML model, a personalized recovery email is generated using the Anthropic API. The prompt is dynamically constructed from the customer's order data and historical spend, with the discount tier tied directly to the model's churn risk score.

**Example output:**
> *Hello Maria, we noticed your order — paid via credit card in 10 installments — was delivered 4 days later than expected. As a valued customer from São Paulo with 6 previous orders, we'd like to offer you a 20% discount on your next purchase. We're sorry for the inconvenience.*

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate olist-analytics
jupyter notebook
```

### Option B — pip

```bash
pip install -r requirements.txt
jupyter notebook
```

### Kaggle credentials

The notebook downloads the dataset automatically via `kagglehub`. You will need a Kaggle account and your API credentials placed at `~/.kaggle/kaggle.json`:

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

Latest notebook run (src workflow) produced the following results:

- **Best classifier:** XGBoost ranked first for low-satisfaction detection (`f1_low_satisfaction=0.422`, `recall_low_satisfaction=0.461`, `roc_auc=0.678`) with a tuned decision threshold of `0.79`.
- **Risk concentration:** Most orders are low risk, but a meaningful at-risk group exists: `89,958` low-risk, `1,507` medium-risk, and `4,358` high-risk orders.
- **Top retention target segment:** Segment `1` (`late-delivery risk | satisfaction risk`) shows the highest low-satisfaction rate at about `65.6%` across `7,604` orders.
- **Most influential model drivers (SHAP):** `delivery_delta`, `total_freight`, and `is_late` are among the strongest predictors, indicating delivery performance has a larger impact than payment dummies alone.
- **Payment behavior signal:** Payment-related features (installments and payment-method dummies) contribute to prediction, but with smaller effect size than delivery-related variables.
- **VIP-weighted training improved business risk handling:** compared to baseline XGBoost, low-satisfaction recall increased (`0.461 -> 0.485`) and VIP false-negative revenue decreased (`126,369.57 -> 120,304.93`).
- **Revenue-at-risk map surfaced code-red geographies for high-value late deliveries:** in the latest run, the largest revenue-at-risk concentrations were in `SP` and `RJ`.
- **Intervention export is production-ready:** `4,358` high-risk delayed-shipping orders were exported to `outputs/intervention/intervention_payload_high_risk_late.csv` with personalization metadata for the email agent.

Note: state-level late-delivery rankings are available in the interactive dashboard section of the notebook and can be exported as a separate table if needed.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy`, `pandasql` |
| Visualization | `plotly`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost`, `shap` |
| GenAI | `anthropic` |
| Environment | `conda`, `jupyter` |

---

## Author

**Laura Romero**
[LinkedIn](https://www.linkedin.com/in/laura-romero-gonzalez) · [GitHub](https://github.com/lauraromerog)

---

## License

This project is for educational and portfolio purposes. The dataset is provided by Olist and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
