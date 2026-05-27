import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔄 Génération de courbes de consommation")
st.markdown("Courbes synthétiques générées par un **GAN** (Generative Adversarial Network) — Arthur")

CACHE_PATH = Path("data/gan_curves.pkl")

@st.cache_data
def load_curves():
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)

if not CACHE_PATH.exists():
    st.error("❌ Fichier gan_curves.pkl introuvable. Lance d'abord run_gan.py en local.")
    st.stop()

data = load_curves()
SEQ_LEN = data['rp']['seq_len']
x_axis  = np.arange(SEQ_LEN) / 2  # en heures (0 à 168)

# ── Sélection ──
type_residence = st.radio(
    "Type de résidence",
    ["🏠 Résidence Principale", "🏖️ Résidence Secondaire"],
    horizontal=True
)
key = "rp" if "Principale" in type_residence else "rs"

real_curves = data[key]['real']
gen_curves  = data[key]['generated']

n_real = st.slider("Nombre de courbes réelles à afficher", 1, len(real_curves), 5)
n_gen  = st.slider("Nombre de courbes générées à afficher", 1, len(gen_curves), 5)

# ── Graphique comparatif ──
st.subheader("📈 Courbes réelles vs générées")

tab1, tab2, tab3 = st.tabs(["Comparaison", "Réelles uniquement", "Générées uniquement"])

with tab1:
    fig = go.Figure()
    for i in range(n_real):
        fig.add_trace(go.Scatter(
            x=x_axis, y=real_curves[i],
            name=f"Réelle {i+1}",
            line=dict(color="gray", width=1),
            opacity=0.6
        ))
    for i in range(n_gen):
        fig.add_trace(go.Scatter(
            x=x_axis, y=gen_curves[i],
            name=f"Générée {i+1}",
            line=dict(width=2, dash="dash"),
        ))
    fig.update_layout(
        title=f"Comparaison réel vs généré — {type_residence}",
        xaxis_title="Heures dans la semaine",
        yaxis_title="Puissance (kW)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    for i in range(n_real):
        fig2.add_trace(go.Scatter(
            x=x_axis, y=real_curves[i],
            name=f"Réelle {i+1}",
            line=dict(width=1.5)
        ))
    fig2.update_layout(
        title="Courbes réelles",
        xaxis_title="Heures", yaxis_title="Puissance (kW)"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    for i in range(n_gen):
        fig3.add_trace(go.Scatter(
            x=x_axis, y=gen_curves[i],
            name=f"Générée {i+1}",
            line=dict(width=1.5, dash="dash")
        ))
    fig3.update_layout(
        title="Courbes générées par le GAN",
        xaxis_title="Heures", yaxis_title="Puissance (kW)"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Statistiques comparatives ──
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

# ── Courbe moyenne ──
st.divider()
st.subheader("📉 Profil moyen de la semaine")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=x_axis, y=real_curves.mean(axis=0),
    name="Moyenne réelle", line=dict(color="black", width=2)
))
fig4.add_trace(go.Scatter(
    x=x_axis, y=gen_curves.mean(axis=0),
    name="Moyenne générée", line=dict(color="royalblue", width=2, dash="dash")
))
fig4.update_layout(
    title="Profil moyen réel vs généré",
    xaxis_title="Heures dans la semaine",
    yaxis_title="Puissance (kW)",
    hovermode="x unified"
)
st.plotly_chart(fig4, use_container_width=True)
