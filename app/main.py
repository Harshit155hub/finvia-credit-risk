from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse
from app.model import predict

app = FastAPI(
    title="Finvia Credit Risk API",
    description="Predicts probability of business default on invoices/loans with SHAP explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_default(body: PredictRequest):
    try:
        result = predict(body.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))