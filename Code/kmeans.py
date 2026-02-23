#%%

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#%%
try:
    # Répertoire du script si exécuté comme fichier
    script_dir = Path(__file__).resolve().parent
except NameError:
    # Fallback pour exécution interactive (Jupyter / Cellules VS Code)
    script_dir = Path.cwd() / "M1EDE" / "Code"

raw_dataset = script_dir / ".." / "data" / "RES2-6-9.csv"
df = pd.read_csv(raw_dataset,
                 sep=';',      # Le séparateur de colonnes
                 decimal='.') # Optionnel : transforme la 2ème colonne en vraie date
print(df.head(), df.info())

#%%
# # --- Convert 'horodate' to datetime with UTC normalization ---
df['horodate'] = pd.to_datetime(df['horodate'], utc=True)  # Convert 'horodate' column to datetime with UTC timezone
df = df.set_index('horodate')  # Set 'horodate' as the index



#%%
# --- Use 'jour' column directly ---
df['jour'] = df.index.date

df_pivot = df.pivot_table(index='ID', columns=df.index, values='valeur', aggfunc='first')
df_pivot = df_pivot.fillna(0)

print(df_pivot.shape, df_pivot.head)

#%%
df_daily = df_pivot.T.resample('D').sum().T
df_daily.head

# #%%
# pca = decomposition.PCA()
# pca.fit(df_pivot)

#%%
# features

# Un jour est considéré comme "inoccupé" si la conso est très faible par rapport à l'habitude
id_mean_conso = df_daily.mean(axis=1)
# Pourcentage de jours de l'année où la maison est "éteinte"
df_daily_norm = df_daily.div(id_mean_conso, axis=0)
#print(df_daily_norm.head)

# Créer des features pour détecter occupation/vacance
df_features = pd.DataFrame()

# Feature 1: Taux d'occupation général (jours avec faible conso par rapport à la moyenne)
df_features['taux_faible_conso'] = ((0 < df_daily_norm) & (df_daily_norm < 0.20)).mean(axis=1)

# Feature 2: Taux d'occupation weekend (samedi/dimanche avec faible conso)
weekend_mask = df_daily_norm.columns.dayofweek >= 5  # 5=samedi, 6=dimanche
df_features['taux_faible_conso_weekend'] = ((0 < df_daily_norm.loc[:, weekend_mask]) & (df_daily_norm.loc[:, weekend_mask] < 0.20)).mean(axis=1)

# Feature 3: Variabilité de la consommation (écart-type normalisé = imprévisibilité)
df_features['variabilite_conso'] = df_daily_norm.std(axis=1)

# Feature 4: Jours inactifs (consommation quasi-nulle)
df_features['taux_jours_inactifs'] = (df_daily_norm < 0.05).mean(axis=1)

# Feature 5: Stabilité consommation (ratio min/max, proche de 1 = stable = occupé)
df_features['stabilite_conso'] = df_daily_norm.min(axis=1) / (df_daily_norm.max(axis=1) + 1e-6)

# --- NOUVELLES FEATURES (Maison Principale vs Secondaire) ---

# Feature 6: Concentration de la consommation 
# (Part de la conso annuelle réalisée sur les 10% des jours les plus intenses. 
# Très élevé pour une maison secondaire par nature.)
def top_10_percent_conso_ratio(row):
    if len(row) == 0: return 0
    sorted_row = row.sort_values(ascending=False)
    top_10_count = max(1, int(len(row) * 0.10))
    return sorted_row.iloc[:top_10_count].sum() / (row.sum() + 1e-6)

df_features['concentration_top10'] = df_daily.apply(top_10_percent_conso_ratio, axis=1)

# Feature 7: Période d'inactivité continue maximale (max jours consécutifs avec faible conso)
def max_consecutive_zeros(row):
    is_inactive = row < 0.15 # Seuil d'inactivité à 15% de la moyenne
    # groupby cumulative sum of negated boolean mask allows counting consecutive Trues
    return is_inactive.groupby((~is_inactive).cumsum()).sum().max()

df_features['max_jours_inactifs_consecutifs'] = df_daily_norm.apply(max_consecutive_zeros, axis=1)

# Feature 8: Saisonnalité (Variance des moyennes mensuelles normalisée)
try:
    df_monthly = df_daily.T.resample('ME').mean().T
    df_features['saisonnalite'] = df_monthly.std(axis=1) / (df_monthly.mean(axis=1) + 1e-6)
except Exception:
    # Fallback pour les anciennes versions de pandas
    df_monthly = df_daily.T.resample('M').mean().T
    df_features['saisonnalite'] = df_monthly.std(axis=1) / (df_monthly.mean(axis=1) + 1e-6)

# Feature 9: Ratio de consommation vacance (juillet-août) par rapport au reste de l'année
vacances_cols = df_daily.columns[df_daily.columns.month.isin([7, 8])]
autres_cols = df_daily.columns[~df_daily.columns.month.isin([7, 8])]

conso_vacances = df_daily[vacances_cols].mean(axis=1) if len(vacances_cols) > 0 else 0
conso_autres = df_daily[autres_cols].mean(axis=1) if len(autres_cols) > 0 else 0

df_features['ratio_vacances_vs_reste'] = conso_vacances / (conso_autres + 1e-6)

# Feature 10: Ratio Week-end / Semaine globale
semaine_mask = df_daily_norm.columns.dayofweek < 5
conso_weekend = df_daily.loc[:, weekend_mask].mean(axis=1) if any(weekend_mask) else 0
conso_semaine = df_daily.loc[:, semaine_mask].mean(axis=1) if any(semaine_mask) else 0

df_features['ratio_weekend_semaine'] = conso_weekend / (conso_semaine + 1e-6)

# Gestion de potentiels NaN créés par des divisions par zéro
df_features = df_features.fillna(0)

print("Shape des features:", df_features.shape)
print(df_features.head())


# #%%
# plt.plot(pca.explained_variance_)
# plt.figure()
# proportion_of_variance = [sum(pca.explained_variance_[0: k])/sum(pca.explained_variance_) for k in range(1, 17472)]
# plt.plot(proportion_of_variance)

# def nb_comp(q):
#   k=0
#   while proportion_of_variance[k] < q:
#     k += 1
#   return k+1
# print(nb_comp(.8))
# print(nb_comp(.9))
# print(nb_comp(.99))
#%%
# --- 1. Préparation des données ---
# Il est important de scaler les données avant d'appliquer K-Means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#%%
# --- 2. Détermination du nombre optimal de clusters (méthode du coude et silhouette) ---
wcss = [] # Within-Cluster Sum of Squares
silhouette_scores = []

# WCSS pour k=1 (silhouette impossible avec 1 cluster)
kmeans_1 = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans_1.fit(scaled_data)
wcss.append(kmeans_1.inertia_)

K_range = range(2, 11)
for i in K_range: # Essayer de 2 à 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters_labels = kmeans.fit_predict(scaled_data)
    wcss.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(scaled_data, clusters_labels)
    silhouette_scores.append(silhouette_avg)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Tracer WCSS
ax1.plot(range(1, 11), wcss, marker='o', linestyle='--')
ax1.set_title('Méthode du coude (WCSS)')
ax1.set_xlabel('Nombre de clusters (k)')
ax1.set_ylabel('WCSS')
ax1.grid(True)

# Tracer Silhouette Score
ax2.plot(K_range, silhouette_scores, marker='s', linestyle='-', color='green')
ax2.set_title('Score de Silhouette par nombre de clusters')
ax2.set_xlabel('Nombre de clusters (k)')
ax2.set_ylabel('Score de Silhouette (plus proche de 1 est mieux)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\nSur la base des graphiques ci-dessus, choisissez une valeur de 'k' optimale.")

# --- 3. Application de l'algorithme K-Means ---
# Remplacez la valeur de 'k' ci-dessous par celle que vous avez choisie après avoir examiné le graphique.
k = 2 # Exemple: nous utilisons k=3. Vous pouvez modifier cette valeur.

kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans_model.fit_predict(scaled_data)

print(f"\nLes données ont été regroupées en {k} clusters.")
print("Premiers 10 clusters assignés :")
print(clusters[:10])

# --- 4. Visualisation des résultats (Exemple avec les 2 premières dimensions) ---
# Si votre dataset a plus de 2 dimensions, la visualisation directe peut être difficile.
# Dans ce cas, vous pourriez utiliser une méthode de réduction de dimension (par exemple, PCA).
if scaled_data.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    for i in range(k):
        plt.scatter(scaled_data[clusters == i, 0], scaled_data[clusters == i, 1], label=f'Cluster {i+1}')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='Centroïdes')
    plt.title(f'Clusters K-Means (k={k}) - Visualisation des 2 premières dimensions')
    plt.xlabel('Dimension 1 (scalée)')
    plt.ylabel('Dimension 2 (scalée)')
    plt.legend()
    plt.grid(True)
    plt.show()
elif scaled_data.shape[1] == 1:
    plt.figure(figsize=(10, 7))
    for i in range(k):
        plt.scatter(scaled_data[clusters == i, 0], np.zeros_like(scaled_data[clusters == i, 0]), label=f'Cluster {i+1}')
    plt.scatter(kmeans_model.cluster_centers_[:, 0], np.zeros_like(kmeans_model.cluster_centers_[:, 0]),
                s=300, c='red', marker='X', label='Centroïdes')
    plt.title(f'Clusters K-Means (k={k}) - Visualisation unidimensionnelle')
    plt.xlabel('Dimension 1 (scalée)')
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("\nLe dataset a des dimensions insuffisantes pour une visualisation standard des clusters.")

#%%
# Utilisation des résultats


print(clusters)
# %%
print([kmeans_model.cluster_centers_[:, i] for i in range(10)])
# %%
print(sum([1 for i in scaled_data[clusters == 0, 0]]), sum([1 for i in scaled_data[clusters == 1, 0]]))
# %%
