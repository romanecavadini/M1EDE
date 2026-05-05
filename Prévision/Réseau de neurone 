import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────

# Charge ton fichier CSV (modifie le chemin si besoin)
df = pd.read_csv("/Users/romanecavadini/Library/Mobile Documents/com~apple~CloudDocs/Ponts/2A/M1EDE/export.csv")

# Définit l'ID du client à analyser (modifie la valeur)
CLIENT_ID = 476866365062

# Filtre pour ne garder que ce client
df = df[df["id"] == CLIENT_ID].copy()

# Conversion de la colonne date
df["horodate"] = pd.to_datetime(df["horodate"], utc=True)
df = df.sort_values("horodate").reset_index(drop=True)

# Extraction de features temporelles
df["heure"]      = df["horodate"].dt.hours
df["jour"]       = df["horodate"].dt.dayofweek
df["mois"]       = df["horodate"].dt.month
df["jour_annee"] = df["horodate"].dt.dayofyear

print("Aperçu des données :")
print(df[["horodate", "valeur", "heure", "jour", "mois"]].head(10))
print(f"\nNombre de points : {len(df)}")
print(f"Consommation moyenne : {df['valeur'].mean():.2f}")

# ─────────────────────────────────────────────
# 2. CONSTRUCTION DES FEATURES ET DE LA CIBLE
# ─────────────────────────────────────────────

feature_cols = ["heure", "jour", "mois", "jour_annee"]

# Ajout de valeurs passées comme features (lag)
for lag in [1, 2, 3]:
    df[f"valeur_lag{lag}"] = df["valeur"].shift(lag)
    feature_cols.append(f"valeur_lag{lag}")

df = df.dropna().reset_index(drop=True)

X = df[feature_cols].values
y = df["valeur"].values

# ─────────────────────────────────────────────
# 3. NORMALISATION
# ─────────────────────────────────────────────

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# ─────────────────────────────────────────────
# 4. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

print(f"\nTrain : {len(X_train)} exemples | Test : {len(X_test)} exemples")

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ─────────────────────────────────────────────
# 5. CONSTRUCTION DU RÉSEAU DE NEURONES
# ─────────────────────────────────────────────

class ReseauConso(nn.Module):
    def __init__(self, n_features):
        super(ReseauConso, self).__init__()
        self.reseau = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.reseau(x)

# Utilise le GPU Apple Silicon si disponible
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nAppareil utilisé : {device}")

model     = ReseauConso(n_features=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────
# 6. ENTRAÎNEMENT
# ─────────────────────────────────────────────

EPOCHS = 50
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Époque {epoch+1}/{EPOCHS} — Loss : {avg_loss:.6f}")

# ─────────────────────────────────────────────
# 7. ÉVALUATION
# ─────────────────────────────────────────────

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t.to(device)).cpu().numpy().flatten()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)

print(f"\n── Résultats ──")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses)
axes[0].set_title("Évolution de la loss")
axes[0].set_xlabel("Époques")
axes[0].set_ylabel("MSE Loss")

axes[1].plot(y_true[:100], label="Réel",   alpha=0.8)
axes[1].plot(y_pred[:100], label="Prédit", alpha=0.8)
axes[1].set_title("Consommation réelle vs prédite (100 premiers points du test)")
axes[1].set_xlabel("Pas de temps")
axes[1].set_ylabel("Consommation")
axes[1].legend()

plt.tight_layout()
plt.savefig("resultats_prediction.png", dpi=150)
plt.show()
print("\nGraphique sauvegardé : resultats_prediction.png")