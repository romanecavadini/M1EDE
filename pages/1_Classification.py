import streamlit as st
import gdown
import shutil
from pathlib import Path as _Path

def _ensure_data():
    _Path("data").mkdir(exist_ok=True)
    if not _Path("data/export.csv").exists():
        gdown.download("https://drive.google.com/uc?id=1aZUOVjMTAhSegI70kPmFjUtPWl644FhT", "data/export.csv", quiet=False)
    if not _Path("data/RES2-6-9.csv").exists():
        shutil.copy("data/export.csv", "data/RES2-6-9.csv")
_ensure_data()
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(layout="wide")
st.title("📊 Classification & Clustering des profils de consommation")

DATA_PATH   = Path("data/RES2-6-9.csv")
LABELS_PATH = Path("data/RES2-6-9-labels.csv")
CACHE_PATH  = Path("data/features_cache.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=',', decimal='.')
    df = df.rename(columns={"id": "pdl_id", "horodate": "datetime", "valeur": "p_kw"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df.dropna(subset=["pdl_id", "datetime", "p_kw"])

@st.cache_data
def load_labels():
    return pd.read_csv(LABELS_PATH, sep=';')

@st.cache_data
def load_features():
    return pd.read_pickle(CACHE_PATH)

@st.cache_data
def prepare_kmeans_features(_df):
    df = _df.copy()
    df_pivot = df.pivot_table(index='pdl_id', columns='datetime', values='p_kw', aggfunc='first').fillna(0)
    df_daily = df_pivot.T.resample('D').sum().T
    id_mean  = df_daily.mean(axis=1)
    df_norm  = df_daily.div(id_mean, axis=0)
    feats    = pd.DataFrame(index=df_daily.index)
    feats['taux_faible_conso']         = ((df_norm > 0) & (df_norm < 0.20)).mean(axis=1)
    weekend_mask = df_norm.columns.dayofweek >= 5
    feats['taux_faible_conso_weekend'] = ((df_norm.loc[:, weekend_mask] > 0) & (df_norm.loc[:, weekend_mask] < 0.20)).mean(axis=1)
    feats['variabilite_conso']         = df_norm.std(axis=1)
    feats['taux_jours_inactifs']       = (df_norm < 0.05).mean(axis=1)
    feats['stabilite_conso']           = df_norm.min(axis=1) / (df_norm.max(axis=1) + 1e-6)
    return feats.fillna(0)

tab1, tab2 = st.tabs(["🔵 Clustering (KMeans)", "🟠 Classification (supervisée)"])

# =============================================================================
# TAB 1 — CLUSTERING KMEANS
# =============================================================================
with tab1:
    st.subheader("KMeans — Regroupement non supervisé des foyers")
    st.markdown("Regroupe les foyers selon leurs patterns de consommation **sans utiliser les labels**.")

    with st.expander("📖 Contexte et choix méthodologiques"):
        st.markdown("""
        **Pourquoi KMeans ?**
        KMeans est un algorithme de clustering non supervisé vu en cours, simple et efficace pour 
        partitionner des données en k groupes homogènes. Il est particulièrement adapté ici car 
        on cherche à regrouper les foyers selon leurs patterns de consommation sans utiliser les labels.

        **Features construites :**
        Plutôt que d'utiliser directement les séries temporelles brutes (trop volumineuses), 
        nous avons construit des features résumant le comportement de chaque foyer :
        - **Taux de faible consommation** : proportion de jours avec une consommation < 20% de la moyenne
        - **Variabilité** : écart-type normalisé de la consommation
        - **Taux de jours inactifs** : proportion de jours quasi sans consommation
        - **Stabilité** : ratio min/max de la consommation journalière

        **Résultats :** Pour k=5, le score de silhouette est de **0.379**, ce qui indique une 
        séparation modérée des clusters. Ce score relativement faible s'explique par la nature 
        continue des comportements de consommation — la frontière entre RP et RS n'est pas toujours nette.
        """)

    df_raw   = load_data()
    df_feats = prepare_kmeans_features(df_raw)
    k        = st.slider("Nombre de clusters (k)", 2, 8, 3)

    # Selectbox client AVANT le bouton
    labels_df_pre = load_labels()
    labels_df_pre['id'] = labels_df_pre['id'].astype(str)
    client_id = st.selectbox("🔍 Recherche par client", labels_df_pre['id'].tolist())

    if st.button("Lancer le clustering"):
        with st.spinner("Clustering en cours..."):
            scaler   = StandardScaler()
            scaled   = scaler.fit_transform(df_feats)
            kmeans   = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            clusters = kmeans.fit_predict(scaled)
            sil      = silhouette_score(scaled, clusters)

            df_feats['cluster'] = clusters
            df_feats.index = df_feats.index.astype(str)

            labels_df = load_labels()
            labels_df['id'] = labels_df['id'].astype(str)
            df_feats_labeled = df_feats.join(labels_df.set_index('id')[['label']], how='left')
            df_feats_labeled['type'] = df_feats_labeled['label'].map({0: '🏠 Principale', 1: '🏖️ Secondaire'})

        st.metric("Score de Silhouette", f"{sil:.3f}", help="Plus proche de 1 = meilleure séparation")

        if client_id in df_feats_labeled.index:
            row = df_feats_labeled.loc[client_id]
            col_a, col_b = st.columns(2)
            col_a.metric("Cluster assigné", f"Cluster {int(row['cluster'])}")
            col_b.metric("Type de résidence", row['type'] if pd.notna(row.get('type')) else "Inconnu")

        st.divider()

        # Scatter avec type de résidence
        fig = px.scatter(
            df_feats_labeled,
            x=df_feats_labeled.columns[0],
            y=df_feats_labeled.columns[1],
            color=df_feats_labeled['cluster'].astype(str),
            symbol='type',
            hover_name=df_feats_labeled.index,
            title=f"Clusters KMeans (k={k}) — forme = type de résidence",
            labels={"color": "Cluster", "symbol": "Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Profil moyen par cluster
        cols_feat = [c for c in df_feats_labeled.columns if c not in ['cluster', 'label', 'type']]
        scaler_mm = MinMaxScaler()
        df_norm2  = pd.DataFrame(
            scaler_mm.fit_transform(df_feats_labeled[cols_feat]),
            columns=cols_feat, index=df_feats_labeled.index
        )
        df_norm2['cluster'] = clusters

        fig2 = go.Figure()
        for i in range(k):
            moyennes = df_norm2[df_norm2['cluster'] == i][cols_feat].mean()
            fig2.add_trace(go.Scatter(y=moyennes.values, x=moyennes.index,
                                      mode='lines+markers', name=f'Cluster {i}'))
        fig2.update_layout(title="Profil moyen normalisé par cluster", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        # Distribution
        counts = pd.Series(clusters).value_counts().sort_index()
        fig3   = px.bar(x=[f"Cluster {i}" for i in counts.index], y=counts.values,
                        title="Nombre de foyers par cluster", labels={"x": "Cluster", "y": "Foyers"})
        st.plotly_chart(fig3, use_container_width=True)

        # Tableau résumé
        st.subheader("📋 Composition des clusters")
        summary = df_feats_labeled.groupby(['cluster', 'type']).size().reset_index(name='Nombre de foyers')
        st.dataframe(summary, use_container_width=True)

# =============================================================================
# TAB 2 — CLASSIFICATION SUPERVISÉE
# =============================================================================
with tab2:
    st.subheader("Classification supervisée — Résidence principale vs secondaire")
    st.markdown("Prédit si un foyer est une **résidence principale (0)** ou **secondaire (1)**.")

    with st.expander("📖 Contexte et choix méthodologiques"):
        st.markdown("""
        **Problématique du déséquilibre de classes**
        Le dataset est très déséquilibré : **86% de résidences principales** vs **14% de secondaires**. 
        Un modèle naïf qui prédit tout en "principale" obtient déjà 86% d'accuracy — ce qui est trompeur.
        C'est pourquoi nous utilisons le **F1-score** et l'**AUC-ROC** comme métriques principales,
        et nous avons introduit un **seuil ajustable** pour mieux détecter les résidences secondaires.

        **Pourquoi abaisser le seuil ?**
        Par défaut, un modèle prédit "secondaire" si la probabilité dépasse 0.5. En abaissant ce seuil 
        (ex. 0.2), on rend le modèle plus sensible aux résidences secondaires au prix de plus de 
        faux positifs. C'est un compromis à ajuster selon l'usage.

        **Comparaison des modèles :**
        - **Régression Logistique** : implémentée from scratch avec descente de gradient. 
          AUC = 1.0 (suspect — possible data leakage car les labels viennent du clustering).
          F1 = 0 à seuil 0.5 (tout prédit en principale).
        - **Random Forest** (Arthur) : accuracy = 0.960, F1 = 0.875, AUC = 0.998. 
          Meilleur modèle grâce à l'ensemble de 4 sous-modèles avec undersampling.

        > ⚠️ Les AUC proches de 1 sont suspects car les labels ont été construits par clustering 
        sur les mêmes données — il peut y avoir une fuite d'information.
        """)

    modele = st.selectbox("Modèle", ["Régression Logistique (Romane)", "Random Forest (Arthur)"])

    # Selectbox client AVANT le bouton
    labels_pre = load_labels()
    client_choisi = st.selectbox("🔍 Client à classifier", labels_pre['id'].tolist())

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit_logistic(X, y, lr=0.01, n_iter=1000):
        beta = np.zeros(X.shape[1])
        for _ in range(n_iter):
            p_hat = sigmoid(X @ beta)
            beta -= lr * (X.T @ (p_hat - y) / len(y))
        return beta

    @st.cache_resource
    def train_logistic(_labels):
        df2  = _labels.copy()
        X    = pd.get_dummies(df2['cluster'], prefix='cluster').values
        y    = df2['label'].values
        X    = np.hstack([np.ones((X.shape[0], 1)), X])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        beta = fit_logistic(X_train, y_train)
        proba = sigmoid(X_test @ beta)
        return y_test, proba

    @st.cache_resource
    def train_rf_arthur(_features, _labels):
        from sklearn.ensemble import RandomForestClassifier
        feat_cols  = [c for c in _features.columns if c != 'pdl_id']
        df_labels  = _labels.copy()
        df_labels  = df_labels.rename(columns={'id': 'pdl_id'})
        df_labels['pdl_id']    = df_labels['pdl_id'].astype(str)
        _features  = _features.copy()
        _features['pdl_id']   = _features['pdl_id'].astype(str)
        df_merged  = _features.merge(df_labels[['pdl_id', 'label']], on='pdl_id', how='inner')
        X          = df_merged[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype('float32')
        y          = df_merged['label'].values
        scaler     = StandardScaler()
        X          = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        idx_pri    = np.where(y_train == 0)[0]
        idx_sec    = np.where(y_train == 1)[0]
        models, all_probas = [], []
        for i, chunk in enumerate(np.array_split(idx_pri, 4)):
            idx = np.concatenate([chunk, idx_sec])
            rf  = RandomForestClassifier(n_estimators=100, random_state=42+i, n_jobs=-1)
            rf.fit(X_train[idx], y_train[idx])
            models.append(rf)
            all_probas.append(rf.predict_proba(X_test))
        mean_probas  = np.mean(all_probas, axis=0)
        proba        = mean_probas[:, 1]
        importances  = np.mean([rf.feature_importances_ for rf in models], axis=0)
        return y_test, proba, feat_cols, importances

    # Slider seuil — toujours visible
    seuil = st.slider(
        "🎚️ Seuil de décision (résidence secondaire)", 0.05, 0.95, 0.5, 0.05,
        help="Baisser le seuil détecte plus de résidences secondaires"
    )

    if st.button("Lancer la classification"):
        labels   = load_labels()
        features = load_features()

        with st.spinner("Entraînement en cours..."):
            if modele == "Régression Logistique (Romane)":
                y_test, proba    = train_logistic(labels)
                show_importance  = False
            else:
                y_test, proba, feat_cols, importances = train_rf_arthur(features, labels)
                show_importance  = True

        y_pred = (proba >= seuil).astype(int)

        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        acc = (y_pred == y_test).mean()

        # ── Résultat pour le client choisi ──
        labels_full = load_labels()
        labels_full['id'] = labels_full['id'].astype(str)
        client_str = str(client_choisi)
        if client_str in labels_full['id'].values:
            row_client = labels_full[labels_full['id'] == client_str].iloc[0]
            cluster_client = row_client['cluster']
            X_all = pd.get_dummies(labels_full['cluster'], prefix='cluster').values
            y_all = labels_full['label'].values
            X_all = np.hstack([np.ones((X_all.shape[0], 1)), X_all])
            def sigmoid_local(z):
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            def fit_local(X, y, lr=0.01, n_iter=1000):
                beta = np.zeros(X.shape[1])
                for _ in range(n_iter):
                    p_hat = sigmoid_local(X @ beta)
                    beta -= lr * (X.T @ (p_hat - y) / len(y))
                return beta
            beta_client = fit_local(X_all, y_all)
            X_client = pd.get_dummies(
                pd.Series([cluster_client], name='cluster'), prefix='cluster'
            ).reindex(columns=pd.get_dummies(labels_full['cluster'], prefix='cluster').columns, fill_value=0).values
            X_client = np.hstack([np.ones((1, 1)), X_client]).astype(float)
            beta_client = beta_client.astype(float)
            proba_client = sigmoid_local(X_client @ beta_client)[0]
            pred_client = "🏖️ Résidence Secondaire" if proba_client >= seuil else "🏠 Résidence Principale"
            vrai_label  = "🏖️ Résidence Secondaire" if row_client['label'] == 1 else "🏠 Résidence Principale"

            st.divider()
            st.subheader(f"🔍 Résultat pour le client {client_choisi}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Prédiction", pred_client)
            col_b.metric("Vrai label", vrai_label)
            col_c.metric("Probabilité secondaire", f"{proba_client:.2%}")
            st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy (globale)", f"{acc:.3f}")
        col2.metric("F1-Score (global)", f"{f1:.3f}")
        col3.metric("AUC-ROC (global)",  f"{auc:.3f}")

        col_left, col_right = st.columns(2)

        with col_left:
            cm     = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               x=["Principale", "Secondaire"],
                               y=["Principale", "Secondaire"],
                               title=f"Matrice de confusion (seuil={seuil})",
                               labels={"x": "Prédit", "y": "Réel"})
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig_roc     = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}", fill='tozeroy'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name="Aléatoire"))
            fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

        if show_importance:
            sorted_idx = np.argsort(importances)[::-1][:10]
            fig_imp    = px.bar(
                x=importances[sorted_idx],
                y=[feat_cols[i] for i in sorted_idx],
                orientation='h', title="Top 10 features importantes (RF)"
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
