# Finvia Credit Risk API

Predictive ML pipeline for estimating the probability that a business will default on an invoice or loan. Built with XGBoost, SHAP, and FastAPI, deployable via Docker.

---

## Project Structure

```
finvia-credit-risk/
├── data/
│   └── generate_data.py        # synthetic dataset generator
├── app/
│   ├── main.py                 # FastAPI app
│   ├── model.py                # inference + SHAP logic
│   └── schemas.py              # request/response models
├── models/                     # saved artifacts after training
├── train.py                    # end-to-end training script
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

```bash
git clone <repo-url>
cd finvia-credit-risk
pip install -r requirements.txt
```

---

## Generating Data & Training

```bash
python data/generate_data.py
python train.py
```

Training outputs saved to `models/`:
- `xgb_model.json`
- `feature_meta.pkl`
- `shap_summary.png`
- `shap_beeswarm.png`
- `confusion_matrix.png`

---

## Running the API

**Locally:**
```bash
uvicorn app.main:app --reload
```

**Docker:**
```bash
docker-compose up --build
```

API docs available at `http://localhost:8000/docs`

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "annual_revenue": 1500000,
    "profit_margin": 0.07,
    "debt_to_equity": 1.2,
    "operating_cash_flow": 95000,
    "years_in_business": 6,
    "avg_invoice_amount": 12000,
    "num_invoices_12m": 45,
    "pct_overdue_invoices": 0.12,
    "avg_days_overdue": 18,
    "payment_disputes": 1,
    "credit_score": 640,
    "num_late_payments": 3,
    "num_credit_accounts": 5,
    "credit_utilization": 0.45,
    "industry_risk_tier": 1
  }'
```

**Response:**
```json
{
  "default_probability": 0.3741,
  "risk_label": "MEDIUM",
  "top_risk_factors": [
    {"feature": "pct_overdue_invoices", "shap_value": 0.412},
    {"feature": "credit_score", "shap_value": -0.318},
    {"feature": "debt_to_equity", "shap_value": 0.201},
    {"feature": "overdue_severity", "shap_value": 0.187},
    {"feature": "payment_risk_index", "shap_value": 0.143}
  ]
}
```

---

## Feature Engineering

Five derived features were added on top of the raw inputs:

| Feature | Formula |
|---|---|
| `revenue_per_invoice` | `annual_revenue / num_invoices_12m` |
| `debt_coverage_ratio` | `operating_cash_flow / debt_to_equity` |
| `overdue_severity` | `pct_overdue_invoices × avg_days_overdue` |
| `credit_health_score` | `credit_score / credit_utilization` |
| `payment_risk_index` | `(late_payments + disputes) / years_in_business` |

These were motivated by standard credit underwriting logic — a business with high overdue % *and* long days outstanding is much riskier than either metric alone.

---

## Model Selection

I tested logistic regression, random forest, and XGBoost on the same train/val split. XGBoost consistently won on PR-AUC, which matters more than ROC-AUC on this imbalanced dataset (~18% default rate). Key hyperparameters:

- `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`
- `scale_pos_weight` set to the negative/positive class ratio to handle imbalance
- L1+L2 regularisation to prevent overfitting on the engineered features

5-fold stratified CV ROC-AUC: **~0.89**

---

## Explainability

SHAP TreeExplainer is used at inference time. Each prediction returns the top 5 features driving that specific risk score — both direction and magnitude. This lets downstream teams (underwriters, ops) understand *why* a business was flagged rather than just that it was.
