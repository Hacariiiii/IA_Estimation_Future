from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()

# ---------------------------
# Charger le modèle au démarrage
# ---------------------------
BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models" / "prophet_ventes_daily.pkl"

data = joblib.load(MODEL_PATH)
model = data["model"]
y_max = data["y_max"]

# ---------------------------
# Classe pour la requête POST
# ---------------------------
class PredictRequest(BaseModel):
    D: int        # nombre de périodes à prédire
    freq: str = "D"  # "D" pour jours ou "M" pour mois

# ---------------------------
# Endpoint de prédiction
# ---------------------------
@app.post("/predict")
async def predict_sales(req: PredictRequest):
    # Créer dataframe futur
    future = model.make_future_dataframe(periods=req.D, freq=req.freq)
    forecast = model.predict(future)

    # Inverse scaling
    forecast["yhat_real"] = np.expm1(forecast["yhat"]) * y_max

    if req.freq.upper() == "M":
        # Si mensuel : sommer les ventes par mois
        forecast['month'] = forecast['ds'].dt.to_period('M')
        monthly_forecast = forecast.groupby('month')['yhat_real'].sum().reset_index()
        monthly_forecast['month'] = monthly_forecast['month'].dt.to_timestamp()
        return monthly_forecast.tail(req.D).to_dict(orient="records")
    else:
        # Si quotidien : renvoyer les ventes par jour
        return forecast[["ds", "yhat_real"]].tail(req.D).to_dict(orient="records")
