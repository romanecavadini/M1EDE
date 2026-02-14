import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

raw_dataset = "../data/RES2-6-9.csv"
df =  pd.read_csv(raw_dataset)


# --- 1. Préparation des données ---
# Il est important de scaler les données avant d'appliquer K-Means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

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