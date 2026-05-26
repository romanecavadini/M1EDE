#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from pathlib import Path

warnings.filterwarnings('ignore') # Pour éviter les warnings d'optimisation de statsmodels

#%%
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd() / "M1EDE" / "Code"

raw_dataset = script_dir / ".." / ".." / "data" / "RES2-6-9.csv"
cache_path  = script_dir / ".." / ".." / "data" / "features_cache.pkl"

# =============================================================================
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES BRUTES
# =============================================================================
COL_PDL = "pdl_id"
COL_DT  = "datetime"
COL_PWR = "p_kw"

raw = pd.read_csv(raw_dataset, sep=';', decimal='.')
raw = raw.rename(columns={"ID": COL_PDL, "id": COL_PDL,
                           "horodate": COL_DT, "valeur": COL_PWR})

raw[COL_DT] = pd.to_datetime(raw[COL_DT], utc=True, errors="coerce")

if raw[COL_DT].dt.tz is None:
    raw[COL_DT] = raw[COL_DT].dt.tz_localize("Europe/Paris")
else:
    raw[COL_DT] = raw[COL_DT].dt.tz_convert("Europe/Paris")

df = raw.dropna(subset=[COL_PDL, COL_DT, COL_PWR]).copy()
df[COL_PWR] = pd.to_numeric(df[COL_PWR], errors="coerce")
df = df.dropna(subset=[COL_PWR])

print(f"✅ Données chargées : {len(df)} lignes, {df[COL_PDL].nunique()} clients")
#%%
# Pour cet exercice, on va extraire les données d'un seul client
# On prend le tout premier ID du dataset
client_id = df[COL_PDL].iloc[0]
print(f"Analyse pour le client ID : {client_id}")

df_client = df[df[COL_PDL] == client_id].copy()
df_client.sort_values(COL_DT, inplace=True)
df_client.set_index(COL_DT, inplace=True)

# S'assurer d'avoir une série temporelle stricte toutes les 30 minutes
# Si des valeurs manquent, on propage la dernière valeur connue (ffill)
ts = df_client[COL_PWR].resample('30min').mean().ffill()

# ==========================================
# 2. SÉPARATION ENTRAÎNEMENT / TEST
# ==========================================
# 1 jour = 48 pas de 30 minutes
# 1 semaine = 48 * 7 = 336 pas de temps
steps_per_week = 336

# On sélectionne 2 semaines consécutives au milieu ou au début
# On décale d'une semaine : on prend la Semaine 2 pour l'entraînement, et la Semaine 3 pour le test
train = ts.iloc[steps_per_week : steps_per_week * 2]  # Semaine 2
test = ts.iloc[steps_per_week * 2 : steps_per_week * 3]   # Semaine 3

print(f"Entraînement (Semaine 2) : du {train.index[0]} au {train.index[-1]}")
print(f"Test (Semaine 3) : du {test.index[0]} au {test.index[-1]}")

# ==========================================
# 3. DÉFINITION ET ENTRAÎNEMENT DU MODÈLE
# ==========================================
# Pour prévoir 1 semaine entière (336 pas) avec des données au pas de 30 minutes, 
# un ARIMA classique (p, d, q) aura tendance à tracer une ligne droite car l'horizon est trop lointain.
# Il est INDISPENSABLE d'utiliser la composante saisonnière (SARIMA).
# On intègre le cycle journalier : s = 48 (car la conso se répète tous les jours)
#%%
print("Entraînement du modèle ARIMA (SARIMA) en cours... (ça peut prendre 1 à 2 minutes)")
# Paramètres ARIMA classiques: p=2, d=0, q=2
# Paramètres Saisonniers: P=1, D=0, Q=1, s=48 (périodicité journalière)
model = ARIMA(train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 48))
model_fit = model.fit()

print("\n--- Résumé du Modèle ---")
# print(model_fit.summary()) # Décommente cette ligne si tu veux voir toutes les stats mathématiques

# ==========================================
# 4. PRÉVISION (FORECAST)
# ==========================================
print("\nCalcul des prévisions pour la Semaine 2...")
predictions = model_fit.forecast(steps=len(test))

# ==========================================
# 5. ÉVALUATION ET VISUALISATION
# ==========================================
# Calcul de l'erreur RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f"Erreur RMSE : {rmse:.2f}")

# Création du graphique
plt.figure(figsize=(15, 6))
plt.plot(train.index, train, label='Semaine 1 (Entraînement)')
plt.plot(test.index, test, label='Semaine 2 (Réalité / Test)')
plt.plot(test.index, predictions, color='red', label='Prédictions ARIMA', linestyle='--')

plt.title(f'Prévision ARIMA d\'une semaine sur la suivante - Client {client_id}')
plt.xlabel('Date')
plt.ylabel('Consommation (kW/h)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
