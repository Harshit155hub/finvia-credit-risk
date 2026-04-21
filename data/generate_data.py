import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 10_000

rng = np.random.default_rng(RANDOM_STATE)


def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    annual_revenue = rng.lognormal(mean=13.5, sigma=1.4, size=n)
    profit_margin = rng.normal(loc=0.08, scale=0.12, size=n).clip(-0.5, 0.6)
    debt_to_equity = rng.exponential(scale=0.8, size=n).clip(0, 6)
    operating_cf = annual_revenue * profit_margin * rng.uniform(0.7, 1.3, size=n)
    years_in_biz = rng.integers(1, 35, size=n)

    avg_invoice_amt = rng.lognormal(mean=9.5, sigma=1.2, size=n)
    num_invoices_12m = rng.integers(2, 120, size=n)
    pct_overdue_invoices = rng.beta(a=2, b=8, size=n)
    avg_days_overdue = rng.exponential(scale=12, size=n)
    payment_disputes = rng.integers(0, 8, size=n)

    credit_score = rng.integers(300, 850, size=n).astype(float)
    num_late_payments = rng.integers(0, 20, size=n)
    num_credit_accounts = rng.integers(1, 15, size=n)
    credit_utilization = rng.beta(a=2, b=3, size=n)

    industry_risk_tier = rng.integers(0, 3, size=n)

    log_odds = (
        -3.5
        - 0.004 * credit_score
        + 0.35 * debt_to_equity
        + 2.8 * pct_overdue_invoices
        + 0.04 * avg_days_overdue
        + 0.12 * num_late_payments
        - 0.6 * profit_margin
        - 0.00000015 * operating_cf
        - 0.02 * years_in_biz
        + 0.3 * industry_risk_tier
        + rng.normal(0, 0.4, size=n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(size=n) < prob_default).astype(int)

    df = pd.DataFrame({
        "annual_revenue": annual_revenue,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "operating_cash_flow": operating_cf,
        "years_in_business": years_in_biz,
        "avg_invoice_amount": avg_invoice_amt,
        "num_invoices_12m": num_invoices_12m,
        "pct_overdue_invoices": pct_overdue_invoices,
        "avg_days_overdue": avg_days_overdue,
        "payment_disputes": payment_disputes,
        "credit_score": credit_score,
        "num_late_payments": num_late_payments,
        "num_credit_accounts": num_credit_accounts,
        "credit_utilization": credit_utilization,
        "industry_risk_tier": industry_risk_tier,
        "default": default,
    })

    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "credit_risk_dataset.csv"
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")