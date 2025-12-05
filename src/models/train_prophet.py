# src/models/train_prophet.py

from prophet import Prophet
import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from src.utils.metrics import mae, rmse, mape

# ----------------------------
# Chemins
# ----------------------------
BASE = Path(__file__).resolve().parents[2]
MONTHLY_DATA_PATH = BASE / 'data' / 'processed' / 'monthly.csv'
MODEL_DIR = BASE / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'table_prophet.pkl'

# ----------------------------
# Étendre les données mensuelles
# ----------------------------
def expand_monthly(df, value_col='y', points_per_month=1, regressors=[]):
	rows = []
	for _, row in df.iterrows():
		month_start = pd.to_datetime(row['ds'])
		for i in range(points_per_month):
			day = month_start + pd.Timedelta(days=int(i * 30 / points_per_month))
			new_row = {
				'ds': day,
				'y': row[value_col] / points_per_month
			}
			for r in regressors:
				new_row[r] = row[r]
			rows.append(new_row)
	return pd.DataFrame(rows)

# ----------------------------
# Script principal
# ----------------------------
if __name__ == "__main__":
	# Vérifier si le fichier existe
	if not MONTHLY_DATA_PATH.exists():
		raise FileNotFoundError(f"Monthly data not found: {MONTHLY_DATA_PATH}")

	# Lire les données
	df = pd.read_csv(MONTHLY_DATA_PATH)
	df['ds'] = pd.to_datetime(df['ds'])

	# Vérifier colonnes numériques
	value_cols = [c for c in df.columns if c not in ['ds', 'is_depense', 'is_revenu']]
	if not value_cols:
		raise ValueError("Aucune colonne numérique trouvée")
	main_col = value_cols[0]

	df = df[['ds', main_col, 'is_depense', 'is_revenu']]
	df = df.rename(columns={main_col: "y"})

	# Normalisation + log pour stabiliser
	y_max = df['y'].max()
	df['y'] = df['y'] / y_max
	df['y'] = np.log1p(df['y'])

	# Étendre les données mensuelles
	df_expanded = expand_monthly(
		df, value_col='y', points_per_month=1,
		regressors=['is_depense','is_revenu']
	)

	# Train / Test split
	train_size = int(len(df_expanded) * 0.8)
	train_df = df_expanded.iloc[:train_size]
	test_df = df_expanded.iloc[train_size:]
	df = df.groupby("ds").sum().reset_index()

	# Modèle Prophet
	m = Prophet(
		yearly_seasonality=True,
		weekly_seasonality=True,
		daily_seasonality=False,
		changepoint_prior_scale=0.1
	)
	
    
	m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
	m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
	m.add_regressor('is_depense')
	m.add_regressor('is_revenu')

	# Entraînement
	m.fit(train_df[['ds','y','is_depense','is_revenu']])

	# Future dataframe
	future = m.make_future_dataframe(periods=len(test_df), freq='D')

	# Ajouter régressors
	future = future.merge(df_expanded[['ds','is_depense','is_revenu']], on='ds', how='left')
	future.fillna(0, inplace=True)

	# Prédiction
	forecast = m.predict(future)

	# ----------------------------
	# Retour à l’échelle réelle
	# ----------------------------
	forecast['yhat_real'] = np.expm1(forecast['yhat']) * y_max
	forecast['yhat_lower_real'] = np.expm1(forecast['yhat_lower']) * y_max
	forecast['yhat_upper_real'] = np.expm1(forecast['yhat_upper']) * y_max

	compare = test_df.merge(forecast[['ds','yhat_real']], on='ds', how='left').dropna()

	y_true_real = np.expm1(compare['y'].values) * y_max
	y_pred_real = compare['yhat_real'].values

	# Métriques
	print("MAE:", mae(y_true_real, y_pred_real))
	print("RMSE:", rmse(y_true_real, y_pred_real))
	print("MAPE:", mape(y_true_real, y_pred_real))

	# Préparer JSON ou POST
	result = forecast[['ds','yhat_real','yhat_lower_real','yhat_upper_real']].to_dict(orient='records')

	# Sauvegarde du modèle
	joblib.dump(m, MODEL_PATH)
	print("Model saved to", MODEL_PATH)
	print("Exemple de prédiction réelle :", result[:5])



