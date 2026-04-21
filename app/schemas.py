from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    annual_revenue: float = Field(..., gt=0)
    profit_margin: float = Field(..., ge=-1.0, le=1.0)
    debt_to_equity: float = Field(..., ge=0)
    operating_cash_flow: float
    years_in_business: int = Field(..., ge=1)
    avg_invoice_amount: float = Field(..., gt=0)
    num_invoices_12m: int = Field(..., ge=1)
    pct_overdue_invoices: float = Field(..., ge=0.0, le=1.0)
    avg_days_overdue: float = Field(..., ge=0)
    payment_disputes: int = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    num_late_payments: int = Field(..., ge=0)
    num_credit_accounts: int = Field(..., ge=1)
    credit_utilization: float = Field(..., ge=0.0, le=1.0)
    industry_risk_tier: int = Field(..., ge=0, le=2)

    class Config:
        json_schema_extra = {
            "example": {
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
                "industry_risk_tier": 1,
            }
        }


class PredictResponse(BaseModel):
    default_probability: float
    risk_label: str
    top_risk_factors: list[dict]