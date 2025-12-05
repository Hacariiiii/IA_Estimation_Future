# src/etl/preprocess_daily.py
import pandas as pd
from pathlib import Path

# Chemins
RAW_PATH = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'finance_powerbi.xlsx'
PROCESSED_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = PROCESSED_DIR / 'monthly.csv'

SHEET_NAME = 'Données'

# ----------------------------
# Chargement du fichier brut
# ----------------------------
def load_raw():
    xls = pd.ExcelFile(RAW_PATH)
    if SHEET_NAME in xls.sheet_names:
        df = pd.read_excel(RAW_PATH, sheet_name=SHEET_NAME)
    else:
        df = pd.read_excel(RAW_PATH, sheet_name=0)
    return df

# ----------------------------
# Détection et parsing des dates
# ----------------------------
def detect_and_parse_date(df):
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'jour' in c.lower()]
    
    if not date_cols:
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.2:
                date_cols.append(c)
                df[c] = parsed
                break
    
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.rename(columns={date_col: 'ds'})
    else:
        raise ValueError('No date column detected')
    
    return df

# ----------------------------
# Nettoyage des colonnes
# ----------------------------
def basic_clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                df[c] = df[c].str.replace(',', '').astype(float)
            except:
                pass
    return df

# ----------------------------
# Exécution principale
# ----------------------------
if __name__ == '__main__':
    df = load_raw()
    df = detect_and_parse_date(df)
    df = basic_clean(df)
    
    # On ne fait plus d'agrégation mensuelle, on garde toutes les lignes
    daily_df = df[['ds','montant','is_depense','is_revenu']]  # ou adapter selon tes colonnes réelles
    daily_df.to_csv(PROCESSED_PATH, index=False)
    print(f'Processed daily series saved to {PROCESSED_PATH}')
