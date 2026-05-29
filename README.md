# Dashboard Data & Énergie — M1EDE, École des Ponts

## Contexte

Ce projet s'inscrit dans le cadre du cours Data & Énergie du parcours Énergie de l'École des Ponts ParisTech. Il s'appuie sur le dataset open data Enedis RES2-6-9kVA, qui contient les courbes de consommation électrique de clients résidentiels avec chauffage électrique, ayant une puissance souscrite comprise entre 6 et 9 kVA, au pas de 30 minutes.

> Les valeurs sont des puissances en kW au pas de 30 minutes. Pour obtenir une énergie en kWh, il faut multiplier par 0,5.

---

## Équipe

| Membre | Contributions |
|--------|--------------|
| Romane | Régression logistique, Random Forest (prévision), Réseau de neurones,dashboard |
| Arthur | KMeans, Neural Network & RF (classification), ARIMA, GAN |
| Salma  | Génération statistique, Visualisation exploratoire, Régression linéaire |

---

## Structure du projetM1EDE/
├── app.py                          # Page d'accueil du dashboard
├── pages/
│   ├── 1_Classification.py         # Clustering KMeans + Classification supervisée
│   ├── 2_Prevision.py              # Prévision de consommation
│   ├── 3_Generation.py             # Génération de courbes synthétiques
│   └── 4_Exploration.py            # Visualisation exploratoire
├── modules/
│   ├── classification/             # Algos de classification
│   ├── prevision/                  # Algos de prévision
│   └── generation/                 # Algos de génération
├── data/
│   ├── RES2-6-9-labels.csv         # Labels RP/RS
│   ├── features_cache.pkl          # Cache features
│   └── gan_curves.pkl              # Courbes GAN pré-générées
└── requirements.txt---

## Modules

### 1. Clustering (non supervisé)
Détection des Résidences Principales (RP) et Secondaires (RS) par KMeans sur des features construites à partir des patterns de consommation journalière.
- k = 5 clusters, score de silhouette = 0.379

### 2. Classification (supervisé)
Prédiction du type de résidence à partir des labels issus du clustering.

| Modèle | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Régression Logistique (Romane) | 0.82 | 0.00 | 1.00 |
| Random Forest (Arthur) | 0.960 | 0.875 | 0.998 |

### 3. Prévision
Prédiction de la consommation sur une semaine, entraîné sur les 2 semaines précédentes.

| Modèle | MAE | RMSE |
|--------|-----|------|
| Random Forest (Romane) | 221.7 | 333.5 |
| Réseau de neurones MLP (Romane) | 234.0 | 342.0 |
| ARIMA (Arthur) | 476.0 | 589.0 |
| Régression Linéaire (Salma) | 6967 | 7894 |

### 4. Génération
Génération de courbes de consommation synthétiques conditionnées au type de résidence.
- GAN (Arthur) : entraîné séparément sur RP et RS, 100 epochs
- Approche statistique (Salma) : profil moyen + bruit gaussien

### 5. Exploration & Visualisation
Visualisation exploratoire des profils de consommation par type de résidence (Salma).

---

## Installation

```bash
git clone https://github.com/romanecavadini/M1EDE.git
cd M1EDE
pip install -r requirements.txt
streamlit run app.py
```

Les données sont téléchargées automatiquement depuis Google Drive au premier lancement.
