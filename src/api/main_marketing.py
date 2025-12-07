from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models" / "prophet_marketing.pkl"
KPI_FILE = BASE / "data" / "processed" / "marketing_kpi_monthly.csv"

data = joblib.load(MODEL_PATH)
model = data["model"]
y_max = data["y_max"]

# --------------------
# KPI ENDPOINT
# --------------------
@app.get("/marketing/kpi")
def get_marketing_kpi():
    df = pd.read_csv(KPI_FILE)

    return {
        "roi_moyen": round(df["Roi_percent"].mean(), 2),
        "taux_engagement_moyen": round(df["taux_engagement_percent"].mean(), 2),
        "ventes_moyennes": round(df["ventes_generees"].mean(), 2),
        "budget_total": round(df["budget"].sum(), 2)
    }

# --------------------
# PREDICTION ENDPOINT
# --------------------
class PredictRequest(BaseModel):
    D: int
    freq: str = "M"

@app.post("/marketing/predict")
async def predict_marketing(req: PredictRequest):
    future = model.make_future_dataframe(periods=req.D, freq=req.freq)
    forecast = model.predict(future)

    forecast["yhat_real"] = np.expm1(forecast["yhat"]) * y_max

    return forecast[["ds", "yhat_real"]].tail(req.D).to_dict(orient="records")
