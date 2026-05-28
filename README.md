# ⚡ Projet Data & Énergie — M1 Parcours Énergie, École des Ponts

## Contexte

Ce projet s'inscrit dans le cadre du cours **Data & Énergie** du parcours Énergie de l'École des Ponts ParisTech. Il s'appuie sur le dataset open data d'Enedis **RES2-6-9kVA**, qui contient les courbes de consommation électrique de clients résidentiels avec chauffage électrique, ayant une puissance souscrite comprise entre 6 et 9 kVA, au pas de 30 minutes.

> ⚠️ Les valeurs sont des **puissances en kW** au pas de 30 minutes. Pour obtenir une énergie en kWh, il faut multiplier par 0,5.

L'objectif est de comprendre et appliquer plusieurs algorithmes d'IA au secteur de l'énergie, à travers 4 modules :
1. **Clustering** — détecter les Résidences Principales (RP) et Secondaires (RS)
2. **Classification** — prédire le type de résidence à partir des consommations
3. **Prévision** — prédire la consommation des jours suivants
4. **Génération** — générer des courbes de consommation synthétiques

---

## Équipe

| Membre | Contributions |
|--------|--------------|
| Romane | Régression logistique, Random Forest (prévision), Réseau de neurones |
| Arthur | KMeans, Neural Network & RF (classification), ARIMA, GAN |
| Salma  | Génération statistique, Visualisation, Régression linéaire |

---

## Structure du projet
M1EDE/
├── app.py                          # Page d'accueil du dashboard
├── pages/
│   ├── 1_Classification.py         # Clustering KMeans + Classification supervisée
│   ├── 2_Prevision.py              # Prévision de consommation
│   ├── 3_Generation.py             # Génération de courbes synthétiques
│   └── 4_Exploration.py            # Visualisation exploratoire (Salma)
├── modules/
│   ├── classification/             # Algos de classification
│   ├── prevision/                  # Algos de prévision
│   └── generation/                 # Algos de génération
├── data/
│   ├── RES2-6-9-labels.csv         # Labels RP/RS
│   ├── features_cache.pkl          # Cache features (build_features.py)
│   └── gan_curves.pkl              # Courbes GAN pré-générées
├── download_data.py                # Téléchargement automatique depuis Google Drive
├── run_gan.py                      # Entraînement du GAN (à lancer en local)
└── requirements.txt---

## Résultats

### 1. Clustering (KMeans)
- **k = 5** clusters, score de silhouette = **0.379**
- Features construites à partir des patterns de consommation journalière (taux d'occupation, variabilité, saisonnalité…)

### 2. Classification supervisée

| Modèle | Accuracy | F1-Score | AUC-ROC | Notes |
|--------|----------|----------|---------|-------|
| Régression Logistique | 0.82 | 0.00 | 1.00 | Seuil abaissé à 0.2 nécessaire (déséquilibre 86%/14%) |
| Random Forest (Arthur) | 0.960 | 0.875 | 0.998 | Meilleur modèle |

> ⚠️ L'AUC-ROC proche de 1 peut suggérer un data leakage — les labels ayant été construits par clustering sur les mêmes données.

### 3. Prévision (par client, sur 1 semaine, entraîné sur 2 semaines)

| Modèle | MAE | RMSE | Notes |
|--------|-----|------|-------|
| Random Forest | 221.7 | 333.5 | Meilleur modèle |
| Réseau de neurones (MLP) | 234.0 | 342.0 | Proche du RF |
| ARIMA | 476.0 | 589.0 | Modèle statistique, moins adapté |
| Régression Linéaire (Salma) | 6967 | 7894 | Travaille sur kWh journaliers (unité différente) |

### 4. Génération
- **GAN** (Arthur) : entraîné séparément sur RP et RS, 100 epochs
- **Approche statistique** (Salma) : profil moyen + bruit gaussien, plus simple et interprétable

---

## Installation

```bash
git clone https://github.com/romanecavadini/M1EDE.git
cd M1EDE
pip install -r requirements.txt
python download_data.py   # Télécharge les données depuis Google Drive
streamlit run app.py
```

---

## Dashboard en ligne

🔗 *(lien Streamlit Cloud à ajouter après déploiement)*
