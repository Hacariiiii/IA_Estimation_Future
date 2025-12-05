# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np

# ----------------------------
# Chemins
# ----------------------------
BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / 'models' / 'table_prophet.pkl'
DATA_PATH = BASE / 'data' / 'processed' / 'monthly.csv'

# ----------------------------
# Classe de requête
# ----------------------------
class PredictionRequest(BaseModel):
    periods: int
    freq: str = 'D'  # 'D' = daily, 'M' = monthly etc.

# ----------------------------
# Lifespan pour charger le modèle au démarrage
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.y_max = 1  # valeur par défaut
    if MODEL_PATH.exists():
        # Charger le modèle
        app.state.model = joblib.load(MODEL_PATH)
        # Récupérer y_max à partir des données
        df = pd.read_csv(DATA_PATH)
        value_cols = [c for c in df.columns if c not in ['ds','is_depense','is_revenu']]
        main_col = value_cols[0]
        app.state.y_max = df[main_col].max()
        print("Model loaded at startup")
    yield

# ----------------------------
# Création de l'app
# ----------------------------
app = FastAPI(lifespan=lifespan)

# ----------------------------
# Route racine
# ----------------------------
@app.get("/")
def root():
    return {"ok": True}

# ----------------------------
# Route de prédiction
# ----------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    model = app.state.model
    y_max = app.state.y_max
    if model is None:
        return {"error": "Model not loaded. Train the model first."}

    # Créer le DataFrame futur
    future = model.make_future_dataframe(periods=req.periods * 30, freq='M')


    # Ajouter les régressors
    try:
        regressors = ['is_depense','is_revenu']
        df_expanded = pd.read_csv(DATA_PATH)
        df_expanded['ds'] = pd.to_datetime(df_expanded['ds'])
        future = future.merge(df_expanded[['ds'] + regressors], on='ds', how='left')
        for r in regressors:
            future[r] = future[r].fillna(0)
    except Exception as e:
        print("Warning: unable to merge regressors:", e)

    # Prédiction
    forecast = model.predict(future)

    # Retour à l'échelle réelle
    yhat_real = np.expm1(forecast['yhat']) * (y_max*10)
    yhat_lower_real = np.expm1(forecast['yhat_lower']) * (y_max*10)
    yhat_upper_real = np.expm1(forecast['yhat_upper']) * (y_max*10)

    # Éviter les valeurs négatives
    yhat_real = yhat_real.clip(lower=0)
    yhat_lower_real = yhat_lower_real.clip(lower=0)
    yhat_upper_real = yhat_upper_real.clip(lower=0)

    # Retourner uniquement les prédictions pour la période demandée
    out = pd.DataFrame({
        'ds': forecast['ds'].dt.strftime('%Y-%m-%d'),
        'yhat': yhat_real,
        'yhat_lower': yhat_lower_real,
        'yhat_upper': yhat_upper_real
    }).tail(req.periods)

    return out.to_dict(orient='records')
