# =========================
# PRÉDICTIONS + VISUALISATION (CORRIGÉ)
# =========================

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# --- sécurité : on repart propre ---
df_model = df_daily.dropna().copy()

# S'assurer que les dates sont au format datetime et triées
df_model["date"] = pd.to_datetime(df_model["date"])
df_model = df_model.sort_values(["pdl_id", "date"])

features = ["lag1", "lag7", "mean_7", "jour_semaine", "mois", "weekend"]

# --- split temporel (évite fuite de données) ---
cutoff = df_model["date"].max() - pd.Timedelta(days=30)

train = df_model[df_model["date"] < cutoff].copy()
test = df_model[df_model["date"] >= cutoff].copy()

# --- entraînement ---
model = LinearRegression()
model.fit(train[features], train["conso_kwh"])

# --- prédictions ---
test["pred"] = model.predict(test[features])

# ==========================================================
# 1. COURBE GLOBALE (Agrégée par jour pour tout le parc)
# ==========================================================
# On additionne la conso de TOUS les clients pour chaque date
df_total = test.groupby("date")[["conso_kwh", "pred"]].sum().reset_index()

plt.figure(figsize=(12, 6))

plt.plot(df_total["date"], df_total["conso_kwh"], label="Réel (Total Parc)", color="tab:blue", linewidth=2)
plt.plot(df_total["date"], df_total["pred"], label="Prédit (Total Parc)", color="tab:orange", linestyle="--", linewidth=2)

plt.title("Consommation Globale : Réel vs Prédit (Somme de tous les clients)")
plt.xlabel("Date")
plt.ylabel("Consommation Totale (kWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# 2. ZOOM SUR UN SEUL CLIENT (Pour vérifier la précision locale)
# ==========================================================
pdl_example = test["pdl_id"].unique()[0] # On prend le premier ID disponible
df_pdl = test[test["pdl_id"] == pdl_example].sort_values("date")

plt.figure(figsize=(12, 5))

plt.plot(df_pdl["date"], df_pdl["conso_kwh"], marker='o', label="Réel", color="tab:blue")
plt.plot(df_pdl["date"], df_pdl["pred"], marker='x', label="Prédit", color="tab:orange")

plt.title(f"Focus Client : {pdl_example} (Réel vs Prédiction)")
plt.xlabel("Date")
plt.ylabel("kWh")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# 3. DISTRIBUTION DES ERREURS
# ==========================================================
test["error"] = test["conso_kwh"] - test["pred"]

plt.figure(figsize=(10, 5))
plt.hist(test["error"], bins=50, color='seagreen', edgecolor='black', alpha=0.7)

plt.axvline(x=0, color='red', linestyle='--', label='Erreur nulle')
plt.title("Distribution des erreurs de prédiction")
plt.xlabel("Erreur (Réel - Prédit) en kWh")
plt.ylabel("Nombre de points")
plt.legend()

plt.tight_layout()
plt.show()
