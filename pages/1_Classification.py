import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("📊 Classification & Clustering des profils de consommation")

DATA_PATH   = Path("data/RES2-6-9.csv")
LABELS_PATH = Path("data/RES2-6-9-labels.csv")
CACHE_PATH  = Path("data/features_cache.pkl")

# ── Chargement données ──
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=',', decimal='.')
    df = df.rename(columns={"id": "pdl_id", "horodate": "datetime", "valeur": "p_kw"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["pdl_id", "datetime", "p_kw"])
    return df

@st.cache_data
def load_labels():
    return pd.read_csv(LABELS_PATH, sep=',')

@st.cache_data
def load_features():
    return pd.read_pickle(CACHE_PATH)

# ── Tabs principaux ──
tab1, tab2 = st.tabs(["🔵 Clustering (KMeans)", "🟠 Classification (supervisée)"])

# =============================================================================
# TAB 1 — CLUSTERING KMEANS (Arthur)
# =============================================================================
with tab1:
    st.subheader("KMeans — Regroupement non supervisé des foyers")
    st.markdown("Regroupe les foyers selon leurs patterns de consommation **sans utiliser les labels**.")

    @st.cache_data
    def prepare_kmeans_features(_df):
        df = _df.copy()
        df['jour'] = df['datetime'].dt.date
        df_pivot = df.pivot_table(index='pdl_id', columns='datetime', values='p_kw', aggfunc='first').fillna(0)
        df_daily = df_pivot.T.resample('D').sum().T

        id_mean = df_daily.mean(axis=1)
        df_norm = df_daily.div(id_mean, axis=0)

        feats = pd.DataFrame(index=df_daily.index)
        feats['taux_faible_conso'] = ((df_norm > 0) & (df_norm < 0.20)).mean(axis=1)
        weekend_mask = df_norm.columns.dayofweek >= 5
        feats['taux_faible_conso_weekend'] = ((df_norm.loc[:, weekend_mask] > 0) & (df_norm.loc[:, weekend_mask] < 0.20)).mean(axis=1)
        feats['variabilite_conso'] = df_norm.std(axis=1)
        feats['taux_jours_inactifs'] = (df_norm < 0.05).mean(axis=1)
        feats['stabilite_conso'] = df_norm.min(axis=1) / (df_norm.max(axis=1) + 1e-6)
        feats = feats.fillna(0)
        return feats

    df_raw = load_data()
    df_feats = prepare_kmeans_features(df_raw)

    k = st.slider("Nombre de clusters (k)", 2, 8, 3)

    if st.button("Lancer le clustering"):
        with st.spinner("Clustering en cours..."):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_feats)

            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            clusters = kmeans.fit_predict(scaled)
            sil = silhouette_score(scaled, clusters)

            df_feats['cluster'] = clusters

        st.metric("Score de Silhouette", f"{sil:.3f}", help="Plus proche de 1 = meilleure séparation")

        # Scatter des 2 premières features
        fig = px.scatter(
            df_feats, x=df_feats.columns[0], y=df_feats.columns[1],
            color=df_feats['cluster'].astype(str),
            title=f"Clusters KMeans (k={k})",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Profil moyen par cluster
        scaler_mm = MinMaxScaler()
        df_norm2 = pd.DataFrame(scaler_mm.fit_transform(df_feats.drop('cluster', axis=1)),
                                columns=df_feats.drop('cluster', axis=1).columns,
                                index=df_feats.index)
        df_norm2['cluster'] = clusters

        fig2 = go.Figure()
        for i in range(k):
            moyennes = df_norm2[df_norm2['cluster'] == i].drop('cluster', axis=1).mean()
            fig2.add_trace(go.Scatter(y=moyennes.values, x=moyennes.index,
                                      mode='lines+markers', name=f'Cluster {i}'))
        fig2.update_layout(title="Profil moyen normalisé par cluster", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        # Distribution
        counts = pd.Series(clusters).value_counts().sort_index()
        fig3 = px.bar(x=[f"Cluster {i}" for i in counts.index], y=counts.values,
                      title="Nombre de foyers par cluster", labels={"x": "Cluster", "y": "Foyers"})
        st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# TAB 2 — CLASSIFICATION SUPERVISÉE
# =============================================================================
with tab2:
    st.subheader("Classification supervisée — Résidence principale vs secondaire")
    st.markdown("Prédit si un foyer est une **résidence principale (0)** ou **secondaire (1)** à partir des labels connus.")

    modele = st.selectbox("Modèle", [
        "Régression Logistique v1 (Romane)",
        "Régression Logistique v2 (Romane)",
        "Random Forest (Arthur)",
    ])

    # ── Fonctions sigmoid et régression logistique ──
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit_logistic(X, y, lr=0.1, n_iter=2000):
        beta = np.zeros(X.shape[1])
        for _ in range(n_iter):
            p_hat = sigmoid(X @ beta)
            beta -= lr * (X.T @ (p_hat - y) / len(y))
        return beta

    @st.cache_resource
    def train_logistic_v1(_features, _labels):
        df2 = _labels.copy()
        X = pd.get_dummies(df2['cluster'], prefix='cluster').values if 'cluster' in df2.columns else pd.get_dummies(df2.iloc[:, -1], prefix='cluster').values
        y = df2['label'].values
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        beta = fit_logistic(X_train, y_train)
        proba = sigmoid(X_test @ beta)
        y_pred = (proba >= 0.5).astype(int)
        return y_test, y_pred, proba

    @st.cache_resource
    def train_logistic_v2(_features, _labels):
        df2 = _labels.copy()
        X = pd.get_dummies(df2['cluster'], prefix='cluster').values if 'cluster' in df2.columns else pd.get_dummies(df2.iloc[:, -1], prefix='cluster').values
        y = df2['label'].values
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        beta = fit_logistic(X_train, y_train, lr=0.01, n_iter=1000)
        proba = sigmoid(X_test @ beta)
        y_pred = (proba >= 0.5).astype(int)
        return y_test, y_pred, proba

    @st.cache_resource
    def train_rf_arthur(_features, _labels):
        from sklearn.ensemble import RandomForestClassifier
        feature_cols = [c for c in _features.columns if c != 'pdl_id']
        df_labels = _labels.rename(columns={"id": "pdl_id"}) if "id" in _labels.columns else _labels
        df_merged = _features.merge(df_labels[['pdl_id', 'label']], on='pdl_id', how='inner')
        X = df_merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype('float32')
        y = df_merged['label'].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        idx_pri = np.where(y_train == 0)[0]
        idx_sec = np.where(y_train == 1)[0]
        models, all_probas = [], []
        for i, chunk in enumerate(np.array_split(idx_pri, 4)):
            idx = np.concatenate([chunk, idx_sec])
            rf = RandomForestClassifier(n_estimators=100, random_state=42+i, n_jobs=-1)
            rf.fit(X_train[idx], y_train[idx])
            models.append(rf)
            all_probas.append(rf.predict_proba(X_test))
        mean_probas = np.mean(all_probas, axis=0)
        y_pred = mean_probas.argmax(axis=1)
        proba = mean_probas[:, 1]
        importances = np.mean([rf.feature_importances_ for rf in models], axis=0)
        return y_test, y_pred, proba, feature_cols, importances

    if st.button("Lancer la classification"):
        labels = load_labels()
        features = load_features() if CACHE_PATH.exists() else None

        with st.spinner("Entraînement en cours..."):
            if modele == "Régression Logistique v1 (Romane)":
                y_test, y_pred, proba = train_logistic_v1(None, labels)
                show_importance = False
            elif modele == "Régression Logistique v2 (Romane)":
                y_test, y_pred, proba = train_logistic_v2(None, labels)
                show_importance = False
            else:
                if features is None:
                    st.error("Cache introuvable. Lance d'abord build_features.py")
                    st.stop()
                y_test, y_pred, proba, feat_cols, importances = train_rf_arthur(features, labels)
                show_importance = True

        # Métriques
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba)
        acc = (y_pred == y_test).mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("F1-Score", f"{f1:.3f}")
        col3.metric("AUC-ROC", f"{auc:.3f}")

        col_left, col_right = st.columns(2)

        # Matrice de confusion
        with col_left:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               x=["Principale", "Secondaire"],
                               y=["Principale", "Secondaire"],
                               title="Matrice de confusion",
                               labels={"x": "Prédit", "y": "Réel"})
            st.plotly_chart(fig_cm, use_container_width=True)

        # Courbe ROC
        with col_right:
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}", fill='tozeroy'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name="Aléatoire"))
            fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance (RF uniquement)
        if show_importance:
            sorted_idx = np.argsort(importances)[::-1][:10]
            fig_imp = px.bar(x=importances[sorted_idx],
                             y=[feat_cols[i] for i in sorted_idx],
                             orientation='h', title="Top 10 features importantes (RF)")
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
