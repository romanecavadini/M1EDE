#%%

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#%%
raw_dataset = Path.cwd() / ".." /"data" / "RES2-6-9.csv"
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

df_features = pd.DataFrame((df_daily_norm < 0.50).mean(axis=1), columns=['taux d\'occupation'])
print(df_features.head)
print(type(df_features))


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
# --- 2. Détermination du nombre optimal de clusters (méthode du coude) ---
wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11): # Essayer de 1 à 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

print("\nSur la base du graphique ci-dessus, choisissez une valeur de 'k' (nombre de clusters) où la courbe forme un 'coude' significatif.")

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
print([kmeans_model.cluster_centers_[:, i] for i in range(365)])
# %%
print(sum([1 for i in scaled_data[clusters == 0, 0]]), sum([1 for i in scaled_data[clusters == 1, 0]]))
# %%
