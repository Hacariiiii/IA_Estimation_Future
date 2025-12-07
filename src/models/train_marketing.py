from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.metrics_marketing import mae, rmse, mape

BASE = Path(__file__).resolve().parents[2]
DATA_FILE = BASE / "data" / "processed" / "marketing_kpi_monthly.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "prophet_marketing.pkl"

df = pd.read_csv(DATA_FILE)

df["ds"] = pd.to_datetime(df["date"])
df["y"] = df["ventes_generees"]

# ----------------------
# 1) OUTLIERS CLEANING
# ----------------------
mean_y = df["y"].mean()
std_y = df["y"].std()
df["y"] = np.where(df["y"] > mean_y + 3 * std_y, mean_y + 3 * std_y, df["y"])

# ----------------------
# 2) SCALING
# ----------------------
y_max = df["y"].max() if df["y"].max() != 0 else 1
df["y_scaled"] = np.log1p(df["y"] / y_max)

# ----------------------
# 3) TRAIN / TEST
# ----------------------
train_size = int(len(df) * 0.85)
train = df.iloc[:train_size][["ds", "y_scaled"]].rename(columns={"y_scaled": "y"})
test = df.iloc[train_size:][["ds", "y_scaled"]].rename(columns={"y_scaled": "y"})

# ----------------------
# 4) MODEL PROPHET
# ----------------------
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.08
)

m.fit(train)

# ----------------------
# 5) VALIDATION
# ----------------------
future = m.make_future_dataframe(periods=len(test), freq="M")
forecast = m.predict(future)

forecast["yhat_real"] = np.expm1(forecast["yhat"]) * y_max

y_true = np.expm1(test["y"]) * y_max
y_pred = forecast["yhat_real"].tail(len(test))

print("\n📊 KPI MARKETING MODEL PERFORMANCE")
print("MAE :", mae(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))

# ----------------------
# 6) SAVE MODEL
# ----------------------
joblib.dump({"model": m, "y_max": y_max}, MODEL_PATH)
print("\n🎉 Modèle KPI Marketing sauvegardé →", MODEL_PATH)
