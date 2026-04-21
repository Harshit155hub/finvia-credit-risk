import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/credit_risk_dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "annual_revenue",
    "profit_margin",
    "debt_to_equity",
    "operating_cash_flow",
    "years_in_business",
    "avg_invoice_amount",
    "num_invoices_12m",
    "pct_overdue_invoices",
    "avg_days_overdue",
    "payment_disputes",
    "credit_score",
    "num_late_payments",
    "num_credit_accounts",
    "credit_utilization",
    "industry_risk_tier",
]
TARGET = "default"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue_per_invoice"] = df["annual_revenue"] / (df["num_invoices_12m"] + 1)
    df["debt_coverage_ratio"] = df["operating_cash_flow"] / (df["debt_to_equity"] + 0.01)
    df["overdue_severity"] = df["pct_overdue_invoices"] * df["avg_days_overdue"]
    df["credit_health_score"] = df["credit_score"] / (df["credit_utilization"] + 0.01)
    df["payment_risk_index"] = (df["num_late_payments"] + df["payment_disputes"]) / (df["years_in_business"] + 1)
    return df


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    engineered = [
        "revenue_per_invoice",
        "debt_coverage_ratio",
        "overdue_severity",
        "credit_health_score",
        "payment_risk_index",
    ]
    all_features = FEATURES + engineered
    X = df[all_features]
    y = df[TARGET]
    return X, y, all_features


def build_model(scale_pos_weight: float):
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


def main():
    X, y, all_features = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_model(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Test PR-AUC:   {average_precision_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    model.save_model(str(MODELS_DIR / "xgb_model.json"))
    joblib.dump({"features": all_features}, MODELS_DIR / "feature_meta.pkl")

    print(f"\nArtifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()