import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# Fonction principale de Prédiction
# ==========================================
def run(df, client_id, target_week_start):
    """
    df : DataFrame complet
    client_id : l'identifiant du client sélectionné
    target_week_start : datetime (le lundi de la semaine à prédire)
    """
    # 1. Copie et conversion des dates pour éviter les conflits
    df_client = df[df['id'] == client_id].copy()
    df_client['horodate'] = pd.to_datetime(df_client['horodate'], utc=True, errors='coerce')
    df_client = df_client.dropna(subset=['horodate']).sort_values('horodate')

    # 2. Encodage client (optionnel ici car mono-client, mais conservé pour les features)
    df_client['id_encoded'] = df_client['id'].astype('category').cat.codes

    # 3. Création des Features temporelles (Lags)
    df_client['lag_1'] = df_client['valeur'].shift(1)
    df_client['lag_2'] = df_client['valeur'].shift(2)
    df_client['lag_48'] = df_client['valeur'].shift(48)      # Même heure veille (si pas de pas fixe)
    df_client['lag_336'] = df_client['valeur'].shift(336)    # Même heure semaine précédente
    
    # 4. Features cycliques
    hour_float = df_client['horodate'].dt.hour + df_client['horodate'].dt.minute / 60
    df_client['sin_hour'] = np.sin(2 * np.pi * hour_float / 24)
    df_client['cos_hour'] = np.cos(2 * np.pi * hour_float / 24)
    df_client['sin_day'] = np.sin(2 * np.pi * df_client['horodate'].dt.dayofweek / 7)
    df_client['cos_day'] = np.cos(2 * np.pi * df_client['horodate'].dt.dayofweek / 7)
    df_client['sin_month'] = np.sin(2 * np.pi * df_client['horodate'].dt.month / 12)
    df_client['cos_month'] = np.cos(2 * np.pi * df_client['horodate'].dt.month / 12)

    df_client = df_client.dropna()

    # 5. Définition des fenêtres temporelles (Semaine cible vs 2 semaines d'avant)
    # On s'assure que target_week_start est bien au format Timestamp UTC
    start_test = pd.to_datetime(target_week_start, utc=True)
    end_test = start_test + pd.Timedelta(days=7)
    start_train = start_test - pd.Timedelta(days=14)
    
    # Split Train / Test
    train_mask = (df_client['horodate'] >= start_train) & (df_client['horodate'] < start_test)
    test_mask = (df_client['horodate'] >= start_test) & (df_client['horodate'] < end_test)
    
    train_df = df_client[train_mask]
    test_df = df_client[test_mask]

    # Vérification si les données existent pour ces périodes
    if train_df.empty or test_df.empty:
        return None  # Pas assez de données pour cette période

    # 6. Features / Target
    features = [
        'id_encoded', 'lag_1', 'lag_2', 'lag_48', 'lag_336',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month'
    ]

    X_train = train_df[features]
    y_train = train_df['valeur']
    X_test = test_df[features]
    y_test = test_df['valeur']

    # 7. Entraînement Modèle
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 8. Prédiction & Évaluation
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "y_true": y_test.values,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "dates": test_df['horodate'].values
    }

# ==========================================
# Interface Streamlit
# ==========================================
st.title("📊 Prévision de Consommation d'Énergie")

# 1. Chargement des données (avec cache pour la fluidité)
path = "/Users/romanecavadini/Library/Mobile Documents/com~apple~CloudDocs/Ponts/2A/M1EDE/export.csv"

@st.cache_data
def load_data(p):
    return pd.read_csv(p)

try:
    df_raw = load_data(path)
except Exception as e:
    st.error(f"Impossible de charger le fichier CSV. Vérifiez le chemin. Erreur : {e}")
    st.stop()

# 2. Widgets de sélection de l'utilisateur
available_clients = df_raw['id'].unique()
selected_client = st.selectbox("👤 Sélectionnez le client :", available_clients)

# 3. Widget de sélection de la date (Semaine)
st.subheader("🗓️ Choisir la semaine à prédire")
chosen_date = st.date_input("Sélectionnez un jour (la prédiction se basera sur sa semaine) :")

# Calcul du lundi de la semaine sélectionnée
start_of_week = pd.to_datetime(chosen_date) - pd.Timedelta(days=chosen_date.weekday())
st.info(f"Semaine de prédiction : du **{start_of_week.strftime('%Y-%m-%d')}** au **{(start_of_week + pd.Timedelta(days=7)).strftime('%Y-%m-%d')}**")
st.caption(f"Période d'entraînement (2 semaines précédentes) : du {(start_of_week - pd.Timedelta(days=14)).strftime('%Y-%m-%d')} au {start_of_week.strftime('%Y-%m-%d')}")

# 4. Lancement de la modélisation
if st.button("🚀 Entraîner le modèle et Prédire"):
    with st.spinner("Modélisation en cours..."):
        results = run(df_raw, selected_client, start_of_week)
        
    if results is None:
        st.error("❌ Données insuffisantes pour ce client sur les périodes demandées (Train de 2 semaines ou Test d'1 semaine manquant).")
    else:
        # Affichage des métriques
        col1, col2 = st.columns(2)
        col1.metric("MAE (Erreur Moyenne Absolue)", f"{results['mae']:.4f}")
        col2.metric("RMSE (Erreur Quadratique Moyenne)", f"{results['rmse']:.4f}")
        
        # Graphique des résultats
        st.subheader(f"📈 Graphique des prévisions - Client {selected_client}")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(results['dates'], results['y_true'], label='Réel (Consommation réelle)', color='blue', alpha=0.7)
        ax.plot(results['dates'], results['y_pred'], label='Prédit (Prévisions)', color='orange', linestyle='--', alpha=0.9)
        ax.set_ylabel("Consommation")
        ax.set_xlabel("Date & Heure")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)