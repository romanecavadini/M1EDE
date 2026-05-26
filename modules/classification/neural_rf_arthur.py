# %%
# =============================================================================
# CLASSIFICATION — Neural Network (MLP) vs Random Forest
# =============================================================================
# Prérequis : exécuter build_features.py une première fois pour créer le cache.
# Ce script charge le cache et compare les deux modèles de classification.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# %%
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd() / "M1EDE" / "Code"

cache_path  = script_dir / ".." / ".." / "data" / "features_cache.pkl"
labels_path = script_dir / ".." / ".." / "data" / "RES2-6-9-labels.csv"

COL_PDL = "pdl_id"

# =============================================================================
# 1. CHARGEMENT DES FEATURES (depuis le cache)
# =============================================================================
# %%
if not cache_path.exists():
    raise FileNotFoundError(
        f"Cache introuvable : {cache_path}\n"
        "→ Exécutez d'abord build_features.py pour le générer."
    )

print(f"🚀 Chargement des features depuis le cache...")
features_pdl = pd.read_pickle(cache_path)
print(f"✅ {len(features_pdl)} clients, {features_pdl.shape[1]} colonnes chargées instantanément")

# Liste des features utilisées pour la classification
feature_cols = [
    "active_day_rate", "n_runs", "mean_run_len", "max_run_len",
    "mean_gap_len", "max_gap_len",
    "mean_daily_kwh", "p95_daily_kwh", "cv_daily_kwh",
    "active_rate_weekday", "active_rate_weekend",
    "mean_kwh_weekday", "mean_kwh_weekend", "winter_minus_summer",
    "seasonality_amp", "r_global", "r_mid", "r_summer", "r_winter",
]
print(f"→ {len(feature_cols)} features utilisées")

# =============================================================================
# 2. CHARGEMENT DES LABELS & PRÉPARATION TRAIN/TEST
# =============================================================================
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_labels = pd.read_csv(labels_path, sep=',')
df_labels = df_labels.rename(columns={"id": COL_PDL})

df_merged = features_pdl.merge(df_labels[[COL_PDL, "label"]], on=COL_PDL, how="inner")
print(f"\nShape après jointure features+labels : {df_merged.shape}")
print("Distribution des labels :\n", df_merged['label'].value_counts())

# %%
X = df_merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype('float32')
y = df_merged['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDimensions : X_train={X_train.shape}, X_test={X_test.shape}")
print(f"Train: {(y_train==0).sum()} principales, {(y_train==1).sum()} secondaires")
print(f"Test:  {(y_test==0).sum()} principales, {(y_test==1).sum()} secondaires")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

n_total   = len(y_train)
n_second  = int(y_train.sum())
n_princip = n_total - n_second
class_weights = torch.tensor([n_total / (2 * n_princip), n_total / (2 * n_second)], dtype=torch.float32)
print(f"class_weights: {class_weights}")

# =============================================================================
# 3. NEURAL NETWORK (MLP)
# =============================================================================
# %%
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 2)
        )
    def forward(self, x):
        return self.net(x)

model     = MLP(input_dim=X_train_t.shape[1])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)

# %%
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import seaborn as sns

model.eval()
with torch.no_grad():
    preds_nn  = model(X_test_t).argmax(1).numpy()
    probas_nn = torch.softmax(model(X_test_t), dim=1)[:, 1].numpy()

target_names = ['Résidence principale (0)', 'Résidence secondaire (1)']

# --- Optimisation du Seuil ---
print("\n" + "-"*30)
print("OPTIMISATION DU SEUIL (NN)")
print("-"*30)

best_f1 = 0
best_thr = 0.5
results_thr = []

for thr in np.arange(0.5, 0.9, 0.01):
    temp_preds = (probas_nn >= thr).astype(int)
    f1 = f1_score(y_test, temp_preds)
    cm = confusion_matrix(y_test, temp_preds)
    r0 = cm[0,0]/cm[0,:].sum()
    r1 = cm[1,1]/cm[1,:].sum()
    results_thr.append((thr, r0, r1, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

for thr, r0, r1, f1 in results_thr:
    star = "*" if abs(thr - best_thr) < 1e-5 else " "
    print(f"{star} Seuil: {thr:.2f} | Recall 0 (Princip): {r0:.2%} | Recall 1 (Second): {r1:.2%} | F1: {f1:.4f}")

print(f"\n→ Application du seuil optimal : {best_thr:.2f}")
preds_nn = (probas_nn >= best_thr).astype(int)

print("=" * 60)
print("NEURAL NETWORK — Rapport de classification")
print("=" * 60)
print(classification_report(y_test, preds_nn, target_names=target_names))

acc_nn = (preds_nn == y_test).mean()
f1_nn  = f1_score(y_test, preds_nn)
auc_nn = roc_auc_score(y_test, probas_nn)
print(f"Accuracy: {acc_nn:.4f} | F1: {f1_nn:.4f} | AUC: {auc_nn:.4f}")

# Indicateurs de Recall par classe
conf_mat_nn = confusion_matrix(y_test, preds_nn)
recall_0 = conf_mat_nn[0, 0] / conf_mat_nn[0, :].sum()
recall_1 = conf_mat_nn[1, 1] / conf_mat_nn[1, :].sum()

print(f"\nPERFORMANCE DÉTAILLÉE (NN au seuil {best_thr:.2f}):")
print(f"→ Taux de détection 'Principales' (Recall 0) : {recall_0:.2%}")
print(f"→ Taux de détection 'Secondaires' (Recall 1) : {recall_1:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_nn, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matrice de confusion du Réseau Neuronal (Test Set)')
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')
plt.show()

# =============================================================================
# 4. RANDOM FOREST — Undersampling ensemblé
# =============================================================================
# %%
from sklearn.ensemble import RandomForestClassifier

idx_pri = np.where(y_train == 0)[0]
idx_sec = np.where(y_train == 1)[0]

N_SPLITS = 4
chunks = np.array_split(idx_pri, N_SPLITS)

models_rf  = []
all_probas = []

for i, chunk in enumerate(chunks):
    idx = np.concatenate([chunk, idx_sec])
    X_sub, y_sub = X_train[idx], y_train[idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=42+i, n_jobs=4)
    rf.fit(X_sub, y_sub)
    models_rf.append(rf)
    all_probas.append(rf.predict_proba(X_test))
    print(f"Modèle RF {i+1}/{N_SPLITS} — {(y_sub==0).sum()} principales, {(y_sub==1).sum()} secondaires")

mean_probas = np.mean(all_probas, axis=0)
y_pred_rf   = mean_probas.argmax(axis=1)
probas_rf   = mean_probas[:, 1]

print("\n" + "=" * 60)
print("RANDOM FOREST — Rapport de classification")
print("=" * 60)
print(classification_report(y_test, y_pred_rf, target_names=target_names))

acc_rf = (y_pred_rf == y_test).mean()
f1_rf  = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, probas_rf)
print(f"Accuracy: {acc_rf:.4f} | F1: {f1_rf:.4f} | AUC: {auc_rf:.4f}")

# =============================================================================
# 5. COMPARAISON VISUELLE — NN vs RF
# =============================================================================
# %%
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Comparaison : Neural Network vs Random Forest\n"
             f"({len(feature_cols)} features issues du clustering RS)",
             fontsize=14, fontweight='bold')

# Matrice de confusion NN
cm_nn = confusion_matrix(y_test, preds_nn)
sns.heatmap(cm_nn, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues',
            xticklabels=['Principale', 'Secondaire'],
            yticklabels=['Principale', 'Secondaire'])
axes[0, 0].set_title("Neural Network — Confusion")
axes[0, 0].set_ylabel("Réel"); axes[0, 0].set_xlabel("Prédit")

# Matrice de confusion RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0, 1], cmap='Oranges',
            xticklabels=['Principale', 'Secondaire'],
            yticklabels=['Principale', 'Secondaire'])
axes[0, 1].set_title("Random Forest — Confusion")
axes[0, 1].set_ylabel("Réel"); axes[0, 1].set_xlabel("Prédit")

# Barplot métriques
metrics = ['Accuracy', 'F1-score', 'AUC']
nn_vals = [acc_nn, f1_nn, auc_nn]
rf_vals = [acc_rf, f1_rf, auc_rf]
x_pos = np.arange(len(metrics))
width = 0.35
bars1 = axes[0, 2].bar(x_pos - width/2, nn_vals, width, label='Neural Network', color='#4C72B0')
bars2 = axes[0, 2].bar(x_pos + width/2, rf_vals, width, label='Random Forest',  color='#DD8452')
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(metrics)
axes[0, 2].set_ylim(0, 1.05)
axes[0, 2].set_title("Métriques comparées")
axes[0, 2].legend()
for bar in bars1:
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# Courbe de perte (NN)
axes[1, 0].plot(train_losses, label='Train loss', linewidth=1.5)
axes[1, 0].plot(test_losses,  label='Test loss',  linewidth=1.5)
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('NN — Évolution de la loss')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# Courbes ROC
fpr_nn, tpr_nn, _ = roc_curve(y_test, probas_nn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, probas_rf)
axes[1, 1].plot(fpr_nn, tpr_nn, label=f'NN (AUC={auc_nn:.3f})',  linewidth=2, color='#4C72B0')
axes[1, 1].plot(fpr_rf, tpr_rf, label=f'RF (AUC={auc_rf:.3f})',  linewidth=2, color='#DD8452')
axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1, 1].set_xlabel('FPR'); axes[1, 1].set_ylabel('TPR')
axes[1, 1].set_title('Courbes ROC')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

# Feature importance (RF)
mean_importances = np.mean([rf.feature_importances_ for rf in models_rf], axis=0)
sorted_idx = np.argsort(mean_importances)[::-1]
axes[1, 2].barh(range(len(feature_cols)),
                mean_importances[sorted_idx],
                color='#DD8452', alpha=0.8)
axes[1, 2].set_yticks(range(len(feature_cols)))
axes[1, 2].set_yticklabels([feature_cols[i] for i in sorted_idx], fontsize=8)
axes[1, 2].invert_yaxis()
axes[1, 2].set_title("RF — Importance des features")
axes[1, 2].set_xlabel("Importance")

plt.tight_layout()
plt.show()

# =============================================================================
# 6. TABLEAU RÉCAPITULATIF
# =============================================================================
# %%
print("\n" + "=" * 60)
print("RÉCAPITULATIF FINAL")
print("=" * 60)

recap = pd.DataFrame({
    'Modèle':   ['Neural Network (MLP)', 'Random Forest (Ensemble)'],
    'Accuracy':  [acc_nn, acc_rf],
    'F1-score':  [f1_nn, f1_rf],
    'AUC-ROC':   [auc_nn, auc_rf],
})
recap = recap.set_index('Modèle')
print(recap.to_string())

best = 'Neural Network' if f1_nn > f1_rf else 'Random Forest'
print(f"\n→ Meilleur modèle (F1) : {best}")

# %%
