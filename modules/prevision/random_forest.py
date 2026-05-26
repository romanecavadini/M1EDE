import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# Chargement des données
# ==========================================

path = "/Users/romanecavadini/Library/Mobile Documents/com~apple~CloudDocs/Ponts/2A/M1EDE/export.csv"

df = pd.read_csv(path)

# ==========================================
# Dates
# ==========================================

df['horodate'] = pd.to_datetime(
    df['horodate'],
    utc=True,
    errors='coerce'
)

df = df.dropna(subset=['horodate'])

# ==========================================
# Tri
# ==========================================

df = df.sort_values(['id', 'horodate'])

# ==========================================
# Encodage client
# ==========================================

df['id_encoded'] = df['id'].astype('category').cat.codes

# ==========================================
# Features temporelles
# ==========================================

# Lags
df['lag_1'] = df.groupby('id')['valeur'].shift(1)
df['lag_2'] = df.groupby('id')['valeur'].shift(2)

# Même heure veille
df['lag_48'] = df.groupby('id')['valeur'].shift(48)

# Même heure semaine précédente
df['lag_336'] = df.groupby('id')['valeur'].shift(336)

# ==========================================
# Features cycliques
# ==========================================

hour_float = (
    df['horodate'].dt.hour +
    df['horodate'].dt.minute / 60
)

df['sin_hour'] = np.sin(2 * np.pi * hour_float / 24)
df['cos_hour'] = np.cos(2 * np.pi * hour_float / 24)

df['sin_day'] = np.sin(
    2 * np.pi * df['horodate'].dt.dayofweek / 7
)

df['cos_day'] = np.cos(
    2 * np.pi * df['horodate'].dt.dayofweek / 7
)

df['sin_month'] = np.sin(
    2 * np.pi * df['horodate'].dt.month / 12
)

df['cos_month'] = np.cos(
    2 * np.pi * df['horodate'].dt.month / 12
)

# ==========================================
# Suppression NaN
# ==========================================

df = df.dropna()

# ==========================================
# Split temporel réaliste
# ==========================================

split_date = "2024-09-01"

train_df = df[df['horodate'] < split_date]
test_df = df[df['horodate'] >= split_date]

print("\n===== PÉRIODES =====")
print(
    "Train :",
    train_df['horodate'].min(),
    "->",
    train_df['horodate'].max()
)

print(
    "Test :",
    test_df['horodate'].min(),
    "->",
    test_df['horodate'].max()
)

# ==========================================
# Features / Target
# ==========================================

features = [
    'id_encoded',
    'lag_1',
    'lag_2',
    'lag_48',
    'lag_336',
    'sin_hour',
    'cos_hour',
    'sin_day',
    'cos_day',
    'sin_month',
    'cos_month'
]

X_train = train_df[features]
y_train = train_df['valeur']

X_test = test_df[features]
y_test = test_df['valeur']

# ==========================================
# Modèle
# ==========================================

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nEntraînement du modèle...")

rf.fit(X_train, y_train)

# ==========================================
# Évaluation globale
# ==========================================

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== PERFORMANCES =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# ==========================================
# Importance des variables
# ==========================================

importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='importance',
    ascending=False
)

print("\n===== IMPORTANCE VARIABLES =====")
print(importance)

# ==========================================
# Visualisation multi-clients
# ==========================================

clients_to_plot = test_df['id'].unique()[:4]

for client_id in clients_to_plot:

    client_data = test_df[
        test_df['id'] == client_id
    ].copy()

    X_client = client_data[features]

    y_real = client_data['valeur'].values

    y_pred_client = rf.predict(X_client)

    plt.figure(figsize=(14,5))

    plt.plot(
        y_real[:500],
        label='Réel'
    )

    plt.plot(
        y_pred_client[:500],
        label='Prédit'
    )

    plt.title(
        f"Prévision consommation - Client {client_id}"
    )

    plt.xlabel("Temps")
    plt.ylabel("Consommation")

    plt.legend()

    plt.show()
def run(df, client_id):
    # utilise le code existant mais avec df en paramètre
    # remplace les pd.read_csv par le df passé en argument
    ...
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse
    }
