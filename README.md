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

**Input features:** `payment_type` (one-hot), `payment_installments`, `price`, `freight_value`, `delivery_delta`, `is_late`, `total_order_value`, `customer_lifetime_orders`

**Models trained:**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble benchmark |
| XGBoost | Primary model |

**Evaluation:** F1-score and AUC-ROC (accuracy avoided due to class imbalance — ~57% 5-star reviews).

**Interpretability:** SHAP values are used to show the directional impact of each feature on predicted satisfaction, going beyond a simple feature importance bar chart.

**Customer segmentation:** K-Means clustering on payment and delivery behavior to identify distinct customer risk profiles (e.g., high-value reliable payers, installment-heavy late-delivery risk).

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

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

---

## Key Findings

*(To be completed after full analysis)*

- [ ] Does payment method (credit card vs boleto) correlate with review score?
- [ ] Do higher installment counts predict lower satisfaction?
- [ ] Is delivery delta the dominant predictor of churn risk?
- [ ] Which Brazilian states show the highest late-delivery rates?

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
