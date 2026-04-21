from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
import shap
import xgboost as xgb

MODELS_DIR = Path(__file__).parent.parent / "models"


@lru_cache(maxsize=1)
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "xgb_model.json"))
    meta = joblib.load(MODELS_DIR / "feature_meta.pkl")
    explainer = shap.TreeExplainer(model)
    return model, meta["features"], explainer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue_per_invoice"] = df["annual_revenue"] / (df["num_invoices_12m"] + 1)
    df["debt_coverage_ratio"] = df["operating_cash_flow"] / (df["debt_to_equity"] + 0.01)
    df["overdue_severity"] = df["pct_overdue_invoices"] * df["avg_days_overdue"]
    df["credit_health_score"] = df["credit_score"] / (df["credit_utilization"] + 0.01)
    df["payment_risk_index"] = (df["num_late_payments"] + df["payment_disputes"]) / (df["years_in_business"] + 1)
    return df


def predict(payload: dict) -> dict:
    model, features, explainer = load_artifacts()

    df = pd.DataFrame([payload])
    df = engineer_features(df)
    df = df[features]

    proba = float(model.predict_proba(df)[0, 1])

    if proba < 0.3:
        risk_label = "LOW"
    elif proba < 0.6:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    shap_values = explainer.shap_values(df)[0]
    factor_list = sorted(
        [{"feature": f, "shap_value": round(float(s), 4)} for f, s in zip(features, shap_values)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "default_probability": round(proba, 4),
        "risk_label": risk_label,
        "top_risk_factors": factor_list[:5],
    }