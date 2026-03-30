# %%
#set up

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# %%
try:
    # Répertoire du script si exécuté comme fichier
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback pour exécution interactive (Jupyter / Cellules VS Code)
    script_dir = Path.cwd() / "M1EDE" / "Code"

raw_dataset = script_dir / ".." / ".." / "data" / "RES2-6-9.csv"
df = pd.read_csv(raw_dataset,
                 sep=';',      # Le séparateur de colonnes
                 decimal='.') # Optionnel : transforme la 2ème colonne en vraie date
print(df.head(), df.info())

# # --- Convert 'horodate' to datetime with UTC normalization ---
df['horodate'] = pd.to_datetime(df['horodate'], utc=True)  # Convert 'horodate' column to datetime with UTC timezone
df = df.set_index('horodate')  # Set 'horodate' as the index

# %%
# --- Pivot df ---
# On pivote la table pour mettre les id en index et les dates en colonnes
df_pivot = df.pivot_table(index='ID', columns=df.index, values='valeur', aggfunc='first')
df_pivot = df_pivot.fillna(0)
print(df_pivot.shape, df_pivot.head())

# %%
# --- Load and merge labels ---
labels_path = script_dir / ".." / ".." / "data" / "RES2-6-9-labels.csv"
# Le fichier contient les colonnes: id,label,cluster
df_labels = pd.read_csv(labels_path, sep=',')
df_labels = df_labels.set_index('id')

# Jointure (inner join pour ne garder que les IDs présents dans les deux tables)
df_merged = df_pivot.join(df_labels[['label']], how='inner')

print(f"Shape après jointure : {df_merged.shape}")
print("Aperçu des labels :\n", df_merged['label'].value_counts())


# %%
# --- Vérification : profil de consommation moyen par label ---
# label=0 : résidence principale | label=1 : résidence secondaire
mean_principale = df_merged[df_merged['label'] == 0].drop(columns=['label']).mean(axis=0)
mean_secondaire = df_merged[df_merged['label'] == 1].drop(columns=['label']).mean(axis=0)

print("Stats — Résidence principale (label=0):")
print(mean_principale.describe())
print("\nStats — Résidence secondaire (label=1):")
print(mean_secondaire.describe())

plt.figure(figsize=(14, 4))
plt.plot(mean_principale.values, label='Résidence principale (0)', alpha=0.8)
plt.plot(mean_secondaire.values, label='Résidence secondaire (1)', alpha=0.8)
plt.title("Profil de consommation moyen — Principale vs Secondaire")
plt.xlabel("Timestamp (index)"); plt.ylabel("Consommation moyenne")
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# %%
# --- Préparation des données ---
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df_merged.drop(columns=['label']).values.astype('float32')
y = df_merged['label'].values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Test split stratifié (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# Pondération des classes pour compenser le déséquilibre
n_total = len(y_train)
n_second   = int(y_train.sum())
n_princip   = n_total - n_second
class_weights = torch.tensor([n_total / (2 * n_princip), n_total / (2 * n_second)], dtype=torch.float32)
print(f"Train: {n_princip} principales, {n_second} secondaires | class_weights: {class_weights}")

# %%
# --- Architecture MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)   # 2 classes : résidence principale (0) ou secondaire (1)
        )
    def forward(self, x):
        return self.net(x)

model     = MLP(input_dim=X_train_t.shape[1])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)

# %%
# --- Boucle d'entraînement ---
EPOCHS = 50
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss   = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_loss   = criterion(test_logits, y_test_t)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        acc = (test_logits.argmax(1) == y_test_t).float().mean().item()
        print(f"Epoch {epoch+1:>3}/{EPOCHS} | Train loss: {loss.item():.4f} | Test loss: {test_loss.item():.4f} | Test acc: {acc:.4f}")

# %%
# --- Évaluation & visualisation ---
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).numpy()

target_names = ['Résidence principale (0)', 'Résidence secondaire (1)']
print(classification_report(y_test_t.numpy(), preds, target_names=target_names))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

cm = confusion_matrix(y_test_t.numpy(), preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
            xticklabels=['Principale', 'Secondaire'],
            yticklabels=['Principale', 'Secondaire'])
axes[0].set_title("Matrice de confusion")
axes[0].set_ylabel("Réel"); axes[0].set_xlabel("Prédit")

axes[1].plot(train_losses, label='Train loss')
axes[1].plot(test_losses,  label='Test loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('Évolution de la loss')
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
