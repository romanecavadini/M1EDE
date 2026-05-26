import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("📈 Prévision de consommation")

@st.cache_data
def load_data():
    df = pd.read_csv("data/export.csv")
    df['horodate'] = pd.to_datetime(df['horodate'], utc=True, errors='coerce')
    df = df.dropna(subset=['horodate'])
    df = df.sort_values(['id', 'horodate'])
    return df

@st.cache_data
def prepare_features(df):
    df = df.copy()
    df['id_encoded'] = df['id'].astype('category').cat.codes
    df['lag_1']   = df.groupby('id')['valeur'].shift(1)
    df['lag_2']   = df.groupby('id')['valeur'].shift(2)
    df['lag_48']  = df.groupby('id')['valeur'].shift(48)
    df['lag_336'] = df.groupby('id')['valeur'].shift(336)
    hour_float = df['horodate'].dt.hour + df['horodate'].dt.minute / 60
    df['sin_hour']  = np.sin(2 * np.pi * hour_float / 24)
    df['cos_hour']  = np.cos(2 * np.pi * hour_float / 24)
    df['sin_day']   = np.sin(2 * np.pi * df['horodate'].dt.dayofweek / 7)
    df['cos_day']   = np.cos(2 * np.pi * df['horodate'].dt.dayofweek / 7)
    df['sin_month'] = np.sin(2 * np.pi * df['horodate'].dt.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['horodate'].dt.month / 12)
    return df.dropna()

FEATURES   = ['id_encoded','lag_1','lag_2','lag_48','lag_336',
               'sin_hour','cos_hour','sin_day','cos_day','sin_month','cos_month']
SPLIT_DATE = "2024-09-01"

@st.cache_resource
def train_random_forest(_df):
    train_df = _df[_df['horodate'] < SPLIT_DATE]
    test_df  = _df[_df['horodate'] >= SPLIT_DATE]
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(train_df[FEATURES], train_df['valeur'])
    return rf, test_df

@st.cache_resource
def train_neural_network(_df):
    train_df = _df[_df['horodate'] < SPLIT_DATE]
    test_df  = _df[_df['horodate'] >= SPLIT_DATE]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train_df[FEATURES].values)
    y_train = scaler_y.fit_transform(train_df['valeur'].values.reshape(-1,1)).flatten()

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=50,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    return mlp, scaler_X, scaler_y, test_df

# ── Interface ──
df_raw  = load_data()
df_feat = prepare_features(df_raw)

modele    = st.selectbox("Modèle", ["Random Forest", "Réseau de neurones"])
client_id = st.selectbox("Client", df_raw['id'].unique())

if st.button("Lancer la prévision"):

    if modele == "Random Forest":
        with st.spinner("Entraînement Random Forest (une seule fois)..."):
            model, test_df = train_random_forest(df_feat)
        client_data = test_df[test_df['id'] == client_id]
        if not client_data.empty:
            y_true = client_data['valeur'].values
            y_pred = model.predict(client_data[FEATURES])

    else:
        with st.spinner("Entraînement réseau de neurones (une seule fois)..."):
            model, scaler_X, scaler_y, test_df = train_neural_network(df_feat)
        client_data = test_df[test_df['id'] == client_id]
        if not client_data.empty:
            y_true = client_data['valeur'].values
            X_scaled = scaler_X.transform(client_data[FEATURES].values)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

    if client_data.empty:
        st.warning("Ce client n'a pas de données dans la période de test.")
    else:
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true[:200], name="Réel",   line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(y=y_pred[:200], name="Prédit", line=dict(color="royalblue", dash="dash")))
        fig.update_layout(title=f"{modele} — Client {client_id}", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("MAE",  f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
