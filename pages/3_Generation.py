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
import pickle
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔄 Génération de courbes de consommation")

DATA_PATH   = Path("data/export.csv")
LABELS_PATH = Path("data/RES2-6-9-labels.csv")
GAN_PATH    = Path("data/gan_curves.pkl")

@st.cache_data
def load_gan_curves():
    with open(GAN_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"id": "pdl_id", "horodate": "datetime", "valeur": "p_kw"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["pdl_id", "datetime", "p_kw"])
    df_daily = df.pivot_table(
        index="pdl_id", columns="datetime", values="p_kw", aggfunc="first"
    ).fillna(0)
    labels = pd.read_csv(LABELS_PATH, sep=';')
    labels['id'] = labels['id'].astype(str)
    df_daily.index = df_daily.index.astype(str)
    df_results = labels.set_index('id')[['label']].copy()
    df_results['prediction_binaire'] = df_results['label']
    return df_daily, df_results

tab1, tab2 = st.tabs([
    "🤖 GAN (Arthur)",
    "📊 Approche Statistique (Salma)"
])

# =============================================================================
# TAB 1 — GAN
# =============================================================================
with tab1:
    st.subheader("GAN — Generative Adversarial Network")
    st.markdown("Courbes synthétiques générées par un **GAN** entraîné séparément sur les résidences principales et secondaires.")

    if not GAN_PATH.exists():
        st.error("❌ Fichier gan_curves.pkl introuvable.")
        st.stop()

    data    = load_gan_curves()
    SEQ_LEN = data['rp']['seq_len']
    x_axis  = np.arange(SEQ_LEN) / 2

    type_residence = st.radio(
        "Type de résidence",
        ["🏠 Résidence Principale", "🏖️ Résidence Secondaire"],
        horizontal=True
    )
    key = "rp" if "Principale" in type_residence else "rs"
    real_curves = data[key]['real']
    gen_curves  = data[key]['generated']

    n_real = st.slider("Courbes réelles à afficher", 1, len(real_curves), 5)
    n_gen  = st.slider("Courbes générées à afficher", 1, len(gen_curves), 5)

    tab1a, tab1b, tab1c = st.tabs(["Comparaison", "Réelles", "Générées"])

    with tab1a:
        fig = go.Figure()
        for i in range(n_real):
            fig.add_trace(go.Scatter(x=x_axis, y=real_curves[i],
                name=f"Réelle {i+1}", line=dict(color="gray", width=1), opacity=0.6))
        for i in range(n_gen):
            fig.add_trace(go.Scatter(x=x_axis, y=gen_curves[i],
                name=f"Générée {i+1}", line=dict(width=2, dash="dash")))
        fig.update_layout(title="Réel vs GAN", xaxis_title="Heures", yaxis_title="kW", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab1b:
        fig2 = go.Figure()
        for i in range(n_real):
            fig2.add_trace(go.Scatter(x=x_axis, y=real_curves[i], name=f"Réelle {i+1}"))
        fig2.update_layout(title="Courbes réelles", xaxis_title="Heures", yaxis_title="kW")
        st.plotly_chart(fig2, use_container_width=True)

    with tab1c:
        fig3 = go.Figure()
        for i in range(n_gen):
            fig3.add_trace(go.Scatter(x=x_axis, y=gen_curves[i],
                name=f"Générée {i+1}", line=dict(dash="dash")))
        fig3.update_layout(title="Courbes générées", xaxis_title="Heures", yaxis_title="kW")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("📊 Statistiques comparatives")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Courbes réelles**")
        st.metric("Moyenne", f"{real_curves.mean():.2f} kW")
        st.metric("Écart-type", f"{real_curves.std():.2f} kW")
        st.metric("Max", f"{real_curves.max():.2f} kW")
    with col2:
        st.markdown("**Courbes générées**")
        st.metric("Moyenne", f"{gen_curves.mean():.2f} kW")
        st.metric("Écart-type", f"{gen_curves.std():.2f} kW")
        st.metric("Max", f"{gen_curves.max():.2f} kW")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=x_axis, y=real_curves.mean(axis=0),
        name="Moyenne réelle", line=dict(color="black", width=2)))
    fig4.add_trace(go.Scatter(x=x_axis, y=gen_curves.mean(axis=0),
        name="Moyenne générée", line=dict(color="royalblue", width=2, dash="dash")))
    fig4.update_layout(title="Profil moyen réel vs GAN",
        xaxis_title="Heures", yaxis_title="kW", hovermode="x unified")
    st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# TAB 2 — Approche Statistique (Salma)
# =============================================================================
with tab2:
    st.subheader("Génération statistique — Profil moyen + bruit gaussien")
    st.markdown("""
    Cette approche génère des courbes synthétiques en ajoutant un **bruit aléatoire gaussien** 
    au profil moyen de consommation observé pour chaque type de résidence.
    C'est une alternative plus simple et interprétable au GAN.
    """)

    with st.spinner("Chargement des données..."):
        df_daily, df_results = load_and_prepare()

    type_stat = st.radio(
        "Type de résidence",
        ["🏠 Résidence Principale", "🏖️ Résidence Secondaire"],
        horizontal=True, key="stat_type"
    )
    consumer_type = 0 if "Principale" in type_stat else 1

    n_curves   = st.slider("Nombre de courbes à générer", 1, 20, 5)
    random_seed = st.slider("Graine aléatoire", 0, 100, 42)

    if st.button("🎲 Générer les courbes"):
        with st.spinner("Génération en cours..."):
            np.random.seed(random_seed)

            # Fusion avec labels
            df_classified = df_daily.merge(
                df_results['prediction_binaire'],
                left_index=True, right_index=True
            )
            df_filtered = df_classified[
                df_classified['prediction_binaire'] == consumer_type
            ].drop(columns=['prediction_binaire'])

            if df_filtered.empty:
                st.error("Aucune donnée pour ce type.")
                st.stop()

            # Agrégation journalière
            df_filtered.columns = pd.to_datetime(df_filtered.columns)
            df_filtered_daily = df_filtered.T.resample("D").sum().T

            mean_profile = df_filtered_daily.mean(axis=0)
            std_profile  = df_filtered_daily.std(axis=0)
            dates        = df_filtered_daily.columns

            # Génération
            synthetic_curves = []
            for _ in range(n_curves):
                curve = mean_profile + np.random.normal(0, std_profile, size=len(mean_profile))
                curve = np.maximum(0, curve)
                synthetic_curves.append(curve)
            synthetic_df = pd.DataFrame(synthetic_curves, columns=dates)

            # Courbes réelles (échantillon)
            real_sample = df_filtered_daily.sample(min(5, len(df_filtered_daily)), random_state=42)

        # Graphique comparaison
        fig_stat = go.Figure()
        for i, row in real_sample.iterrows():
            fig_stat.add_trace(go.Scatter(
                x=list(range(len(row))), y=row.values,
                name=f"Réelle", line=dict(color="gray", width=1), opacity=0.5,
                showlegend=(i == real_sample.index[0])
            ))
        for i, row in synthetic_df.iterrows():
            fig_stat.add_trace(go.Scatter(
                x=list(range(len(row))), y=row.values,
                name=f"Synthétique {i+1}",
                line=dict(width=2, dash="dash")
            ))
        fig_stat.update_layout(
            title=f"Courbes réelles vs synthétiques — {type_stat}",
            xaxis_title="Jours", yaxis_title="Consommation (kWh)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_stat, use_container_width=True)

        # Profil moyen
        fig_mean = go.Figure()
        fig_mean.add_trace(go.Scatter(
            y=mean_profile.values, name="Profil moyen réel",
            line=dict(color="black", width=2)
        ))
        fig_mean.add_trace(go.Scatter(
            y=synthetic_df.mean(axis=0).values, name="Profil moyen synthétique",
            line=dict(color="royalblue", width=2, dash="dash")
        ))
        fig_mean.update_layout(
            title="Profil moyen réel vs synthétique",
            xaxis_title="Jours", yaxis_title="Consommation (kWh)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_mean, use_container_width=True)

        # Tableau indicateurs
        st.subheader("📋 Indicateurs statistiques comparatifs")
        indicators = pd.DataFrame({
            "": ["Réel", "Synthétique"],
            "Moyenne totale": [
                df_filtered_daily.sum(axis=1).mean(),
                synthetic_df.sum(axis=1).mean()
            ],
            "Écart-type": [
                df_filtered_daily.sum(axis=1).std(),
                synthetic_df.sum(axis=1).std()
            ],
            "Min": [
                df_filtered_daily.sum(axis=1).min(),
                synthetic_df.sum(axis=1).min()
            ],
            "Max": [
                df_filtered_daily.sum(axis=1).max(),
                synthetic_df.sum(axis=1).max()
            ],
            "RMSE (vs moyenne réelle)": [
                float(np.sqrt(np.mean((df_filtered_daily.mean(axis=0).values - df_filtered_daily.mean(axis=0).values)**2))),
                float(np.sqrt(np.mean((synthetic_df.mean(axis=0).values - df_filtered_daily.mean(axis=0).values)**2)))
            ],
        }).set_index("")
        st.dataframe(indicators.style.format("{:.2f}"), use_container_width=True)
