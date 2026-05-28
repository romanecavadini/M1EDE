import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH  = Path("data/export.csv")
CACHE_PATH = Path("data/rf_prepared.pkl")

FEATURES = [
    'id_encoded', 'lag_1', 'lag_2', 'lag_48', 'lag_336',
    'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month'
]

st.title("📈 Prévision de consommation")

with st.expander("📖 Contexte et choix méthodologiques"):
    st.markdown("""
    **Objectif :** Prédire la consommation d'un client sur une semaine cible, 
    en s'entraînant sur les 2 semaines précédentes.

    **Features utilisées (RF et MLP) :**
    - Lags temporels : consommation à t-1, t-2, t-48 (même heure la veille), t-336 (même heure la semaine précédente)
    - Features cycliques : sin/cos de l'heure, du jour de la semaine et du mois

    **Comparaison des modèles :**

    | Modèle | MAE | RMSE | Commentaire |
    |--------|-----|------|-------------|
    | Random Forest | 221.7 | 333.5 | Meilleur modèle — capture bien les non-linéarités |
    | Réseau de neurones (MLP) | 234.0 | 342.0 | Proche du RF, sklearn MLPRegressor |
    | ARIMA | 476.0 | 589.0 | Modèle statistique, moins adapté aux longues séries |
    | Régression Linéaire (Salma) | 6967 | 7894 | Travaille sur kWh journaliers (unité différente) |

    **Pourquoi le Random Forest est meilleur ?**
    Il capture mieux les non-linéarités et les interactions entre features que la régression linéaire.
    ARIMA est limité car il ne modélise pas bien la saisonnalité complexe des séries de consommation.
    Le MLP est comparable au RF mais plus lent à entraîner.

    > ⚠️ La régression linéaire de Salma n'est pas directement comparable aux autres car elle 
    prédit la **consommation journalière totale** (kWh/jour) et non la puissance au pas de 30 min.
    """)

# ── Cache disque : chargé une fois pour toutes ──
@st.cache_resource
def load_data():
    if CACHE_PATH.exists():
        return pd.read_pickle(CACHE_PATH)
    df = pd.read_csv(DATA_PATH)
    df['horodate'] = pd.to_datetime(df['horodate'], utc=True, errors='coerce')
    df = df.dropna(subset=['horodate']).sort_values(['id', 'horodate'])
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
    df = df.dropna()
    df.to_pickle(CACHE_PATH)
    return df

# ── Modèles ──
def run_random_forest(df, client_id, start_of_week):
    df_client   = df[df['id'] == client_id]
    start_test  = pd.to_datetime(start_of_week, utc=True)
    end_test    = start_test + pd.Timedelta(days=7)
    start_train = start_test - pd.Timedelta(days=14)
    train_df = df_client[(df_client['horodate'] >= start_train) & (df_client['horodate'] < start_test)]
    test_df  = df_client[(df_client['horodate'] >= start_test)  & (df_client['horodate'] < end_test)]
    if train_df.empty or test_df.empty:
        return None
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(train_df[FEATURES], train_df['valeur'])
    y_pred = rf.predict(test_df[FEATURES])
    return {
        "y_true": test_df['valeur'].values,
        "y_pred": y_pred,
        "mae":    mean_absolute_error(test_df['valeur'], y_pred),
        "rmse":   np.sqrt(mean_squared_error(test_df['valeur'], y_pred)),
        "dates":  test_df['horodate'].values
    }

def run_mlp(df, client_id, start_of_week):
    df_client   = df[df['id'] == client_id]
    start_test  = pd.to_datetime(start_of_week, utc=True)
    end_test    = start_test + pd.Timedelta(days=7)
    start_train = start_test - pd.Timedelta(days=14)
    train_df = df_client[(df_client['horodate'] >= start_train) & (df_client['horodate'] < start_test)]
    test_df  = df_client[(df_client['horodate'] >= start_test)  & (df_client['horodate'] < end_test)]
    if train_df.empty or test_df.empty:
        return None
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train  = scaler_X.fit_transform(train_df[FEATURES].values)
    y_train  = scaler_y.fit_transform(train_df['valeur'].values.reshape(-1,1)).flatten()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=50, random_state=42)
    mlp.fit(X_train, y_train)
    X_test = scaler_X.transform(test_df[FEATURES].values)
    y_pred = scaler_y.inverse_transform(mlp.predict(X_test).reshape(-1,1)).flatten()
    return {
        "y_true": test_df['valeur'].values,
        "y_pred": y_pred,
        "mae":    mean_absolute_error(test_df['valeur'], y_pred),
        "rmse":   np.sqrt(mean_squared_error(test_df['valeur'], y_pred)),
        "dates":  test_df['horodate'].values
    }

def run_arima(df, client_id, start_of_week):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')
    df_client = df[df['id'] == client_id].copy()
    df_client = df_client.set_index('horodate')[['valeur']]
    ts = df_client['valeur'].resample('30min').mean().ffill()
    start_test  = pd.to_datetime(start_of_week, utc=True)
    end_test    = start_test + pd.Timedelta(days=7)
    start_train = start_test - pd.Timedelta(days=14)
    train = ts[(ts.index >= start_train) & (ts.index < start_test)]
    test  = ts[(ts.index >= start_test)  & (ts.index < end_test)]
    if len(train) < 10 or len(test) == 0:
        return None
    model      = ARIMA(train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 48))
    model_fit  = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    return {
        "y_true": test.values,
        "y_pred": predictions.values,
        "mae":    mean_absolute_error(test.values, predictions.values),
        "rmse":   np.sqrt(mean_squared_error(test.values, predictions.values)),
        "dates":  test.index
    }


def run_linear_regression(df, client_id, start_of_week):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Préparation des données journalières
    df_client = df[df["id"] == client_id].copy()
    df_client = df_client.set_index("horodate")[["valeur"]]
    df_daily  = df_client["valeur"].resample("D").sum().reset_index()
    df_daily.columns = ["date", "conso_kwh"]
    df_daily["pdl_id"]       = client_id
    df_daily["jour_semaine"] = df_daily["date"].dt.dayofweek
    df_daily["mois"]         = df_daily["date"].dt.month
    df_daily["weekend"]      = df_daily["date"].dt.weekday // 5
    df_daily["lag1"]         = df_daily["conso_kwh"].shift(1)
    df_daily["lag7"]         = df_daily["conso_kwh"].shift(7)
    df_daily["mean_7"]       = df_daily["conso_kwh"].rolling(window=7, min_periods=1).mean()
    df_daily = df_daily.dropna()

    start_test  = pd.to_datetime(start_of_week, utc=True)
    end_test    = start_test + pd.Timedelta(days=7)
    cutoff      = start_test

    train = df_daily[df_daily["date"] < cutoff]
    test  = df_daily[(df_daily["date"] >= cutoff) & (df_daily["date"] < end_test)]

    if train.empty or test.empty:
        return None

    features = ["lag1", "lag7", "mean_7", "jour_semaine", "mois", "weekend"]
    model    = LinearRegression()
    model.fit(train[features], train["conso_kwh"])
    y_pred   = model.predict(test[features])
    y_true   = test["conso_kwh"].values

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mae":    mean_absolute_error(y_true, y_pred),
        "rmse":   np.sqrt(mean_squared_error(y_true, y_pred)),
        "dates":  test["date"].values
    }

# ── Interface ──
df_prepared = load_data()

modele          = st.selectbox("Modèle", ["Random Forest", "Réseau de neurones", "ARIMA", "Régression Linéaire (Salma)"])
selected_client = st.selectbox("👤 Client", df_prepared['id'].unique())

st.subheader("🗓️ Semaine à prédire")
chosen_date   = st.date_input("Sélectionnez un jour")
start_of_week = pd.to_datetime(chosen_date) - pd.Timedelta(days=chosen_date.weekday())
end_of_week   = start_of_week + pd.Timedelta(days=7)
st.info(f"Semaine cible : du **{start_of_week.strftime('%Y-%m-%d')}** au **{end_of_week.strftime('%Y-%m-%d')}** — entraîné sur les 2 semaines précédentes")

if st.button("🚀 Lancer la prévision"):
    with st.spinner("Calcul en cours..."):
        if modele == "Random Forest":
            results = run_random_forest(df_prepared, selected_client, start_of_week)
        elif modele == "Réseau de neurones":
            results = run_mlp(df_prepared, selected_client, start_of_week)
        elif modele == "ARIMA":
            results = run_arima(df_prepared, selected_client, start_of_week)
        else:
            results = run_linear_regression(df_prepared, selected_client, start_of_week)

    if results is None:
        st.error("❌ Données insuffisantes pour ce client sur cette période. Essaie une autre semaine.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("MAE",  f"{results['mae']:.4f}")
        col2.metric("RMSE", f"{results['rmse']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['dates'], y=results['y_true'],
            name="Réel", line=dict(color="black", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=results['dates'], y=results['y_pred'],
            name="Prédit", line=dict(color="royalblue", dash="dash")
        ))
        fig.update_layout(
            title=f"{modele} — Client {selected_client}",
            xaxis_title="Date", yaxis_title="Consommation (kWh)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
