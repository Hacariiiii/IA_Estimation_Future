import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
INPUT = BASE / "data" / "raw" / "marketing.csv"
OUTPUT = BASE / "data" / "processed" / "marketing_kpi_monthly.csv"

df = pd.read_csv(INPUT)
df = df.dropna()

# Création de la date mensuelle
df["date"] = pd.to_datetime(df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01")

# Agrégation mensuelle des KPI
df_kpi = df.groupby("date").agg({
    "ventes_generees": "mean",          # ✅ ventes moyennes
    "Roi_percent": "mean",              # ✅ ROI moyen
    "taux_engagement_percent": "mean",  # ✅ Taux engagement moyen
    "budget": "sum"                     # Budget total
}).reset_index()

df_kpi.to_csv(OUTPUT, index=False)
print("✅ KPI Marketing mensuels créés →", OUTPUT)
