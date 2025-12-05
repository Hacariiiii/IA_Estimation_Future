import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW_FILE = BASE / "data" / "raw" / "ventes_cnv.csv"
PROCESSED_FILE = BASE / "data" / "processed" / "daily_ventes.csv"

def preprocess_ventes_daily():

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw ventes file: {RAW_FILE}")

    print(f"Reading {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    # Check required columns
    required_cols = ['date_vente', 'montant_total']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Parse dates safely
    df['date_vente'] = pd.to_datetime(df['date_vente'], errors='coerce')
    df = df.dropna(subset=['date_vente'])
    df['montant_total'] = pd.to_numeric(df['montant_total'], errors='coerce').fillna(0)

    # Aggregate per day
    daily = (
        df.groupby(pd.Grouper(key='date_vente', freq='D'))
          .agg({'montant_total': 'sum'})
          .reset_index()
    )

    daily = daily.rename(columns={'date_vente': 'ds', 'montant_total': 'y'})
    daily = daily.sort_values('ds').reset_index(drop=True)

    # Ensure directory exists
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_FILE, index=False)

    print(f"Daily file saved: {PROCESSED_FILE}")
    print(f"Rows: {len(daily)} / Date range: {daily['ds'].min()} -> {daily['ds'].max()}")

    return daily


if __name__ == "__main__":
    preprocess_ventes_daily()
