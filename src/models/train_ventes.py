from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from prophet.diagnostics import cross_validation, performance_metrics
from src.utils.metrics_ventes import mae, rmse, mape

BASE = Path(__file__).resolve().parents[2]
DATA_FILE = BASE / "data" / "processed" / "daily_ventes.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "prophet_ventes_daily.pkl"

if __name__ == "__main__":

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["ds"] = pd.to_datetime(df["ds"])

    # --------------------------------------
    # 1) CLEANING DATA
    # --------------------------------------

    # Remove negative or crazy outliers (> mean + 3*std)
    mean_y = df["y"].mean()
    std_y = df["y"].std()
    df["y"] = np.where(df["y"] > mean_y + 3 * std_y, mean_y + 3 * std_y, df["y"])

    # Fill missing days (important for prophet)
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq='D')
    df = df.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
    df["y"] = df["y"].fillna(0)

    # --------------------------------------
    # 2) SCALING
    # --------------------------------------
    y_max = df["y"].max() if df["y"].max() != 0 else 1.0
    df["y_scaled"] = np.log1p(df["y"] / y_max)

    # --------------------------------------
    # 3) TRAIN / TEST
    # --------------------------------------
    train_size = int(len(df) * 0.85)
    train = df.iloc[:train_size][["ds", "y_scaled"]]
    test = df.iloc[train_size:][["ds", "y_scaled"]]

    # Prophet wants column name exactly "y"
    train = train.rename(columns={"y_scaled": "y"})
    test = test.rename(columns={"y_scaled": "y"})

    # --------------------------------------
    # 4) MODEL 🔥 (BEST CONFIG)
    # --------------------------------------
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,      # more sensitive to trend changes
        seasonality_prior_scale=10,
        interval_width=0.90
    )

    # Monthly seasonality (very important!)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=6)

    # --------------------------------------
    # FIT
    # --------------------------------------
    print("Training Prophet model…")
    m.fit(train)

    # --------------------------------------
    # 5) FORECAST on TEST
    # --------------------------------------
    future = m.make_future_dataframe(periods=len(test), freq='D')
    forecast = m.predict(future)

    # Inverse scaling
    forecast["yhat_real"] = np.expm1(forecast["yhat"]) * y_max

    comp = test.merge(forecast[["ds", "yhat_real"]], on="ds", how="left").dropna()

    y_true = np.expm1(comp["y"]) * y_max
    y_pred = comp["yhat_real"]

    # --------------------------------------
    # 6) METRICS
    # --------------------------------------
    print("\n📊 MODEL PERFORMANCE (DAILY)")
    print("MAE :", mae(y_true, y_pred))
    print("RMSE:", rmse(y_true, y_pred))
    print("MAPE:", mape(y_true, y_pred))

    # --------------------------------------
    # 7) CROSS VALIDATION (90 days horizon)
    # --------------------------------------
    print("\nRunning cross-validation…")
    df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='90 days')
    df_p = performance_metrics(df_cv)

    print("\n📈 CROSS VALIDATION RESULTS:")
    print(df_p)

    # --------------------------------------
    # 8) SAVE MODEL + y_max
    # --------------------------------------
    joblib.dump(
        {"model": m, "y_max": y_max},
        MODEL_PATH
    )

    print("\n🎉 Model saved successfully →", MODEL_PATH)
