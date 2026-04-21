# Finvia Credit Risk API

A machine learning-powered API that predicts the probability of default for small businesses based on financial and behavioral data.

This project simulates a real-world fintech credit risk system where decisions must be both accurate and explainable. Along with prediction, the API returns the key factors influencing risk using SHAP, making it useful for analysts and underwriting teams.

---

## Why I Built This

Traditional credit scoring systems often act as black boxes. I wanted to build a system that not only predicts risk but also explains the reasoning behind it.

Through this project, I explored:

* Real-world credit risk modeling concepts
* Handling imbalanced datasets
* Feature engineering using financial logic
* Making ML models interpretable using SHAP

---

## Tech Stack

* Python
* FastAPI
* XGBoost
* SHAP
* Docker
* Pandas, NumPy, Scikit-learn

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

* `xgb_model.json`
* `feature_meta.pkl`
* `shap_summary.png`
* `shap_beeswarm.png`
* `confusion_matrix.png`

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

API docs available at:
http://localhost:8000/docs

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

Five derived features were created to improve model performance:

| Feature               | Formula                                                      |
| --------------------- | ------------------------------------------------------------ |
| `revenue_per_invoice` | `annual_revenue / num_invoices_12m`                          |
| `debt_coverage_ratio` | `operating_cash_flow / debt_to_equity`                       |
| `overdue_severity`    | `pct_overdue_invoices × avg_days_overdue`                    |
| `credit_health_score` | `credit_score / credit_utilization`                          |
| `payment_risk_index`  | `(num_late_payments + payment_disputes) / years_in_business` |

These features are based on financial reasoning. For example, a business with both a high percentage of overdue invoices and long overdue durations is significantly riskier than considering either factor alone.

---

## Model Selection

Multiple models were tested, including logistic regression, random forest, and XGBoost. XGBoost performed the best based on PR-AUC, which is more suitable for imbalanced datasets (~5% default rate).

Key configuration:

* `n_estimators = 400`
* `max_depth = 5`
* `learning_rate = 0.05`
* `scale_pos_weight` used to handle class imbalance
* L1 and L2 regularization to reduce overfitting

5-fold stratified cross-validation ROC-AUC: **~0.74**

---

## Explainability

SHAP TreeExplainer is used during inference to interpret predictions. For each request, the API returns the top contributing features along with their impact.

This makes the model transparent and usable in real-world scenarios where understanding the reason behind a decision is critical.

---

## Future Improvements

* Integrate real-world financial datasets
* Add authentication and rate limiting to the API
* Build a frontend dashboard for visualization
* Deploy the system on cloud platforms

