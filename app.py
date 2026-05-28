import streamlit as st
import subprocess

subprocess.run(["python", "download_data.py"])

st.set_page_config(
    page_title="Data & Énergie",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard Data & Énergie")
st.markdown("**M1 Parcours Énergie — École des Ponts ParisTech**")

st.divider()

st.markdown("""
### Contexte du projet

Ce dashboard présente les résultats de notre projet de cours **Data & Énergie**, 
réalisé dans le cadre du parcours Énergie de l'École des Ponts ParisTech.

Nous travaillons sur le dataset open data Enedis **RES2-6-9kVA** : des courbes de 
consommation électrique de **500 clients résidentiels** avec chauffage électrique, 
relevées au **pas de 30 minutes** sur environ un an.

L'objectif est de détecter si chaque foyer est une **Résidence Principale (RP)** 
ou une **Résidence Secondaire (RS)**, puis de prévoir et générer des courbes de consommation.
""")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Foyers analysés", "500")
col2.metric("Période", "2023 – 2024")
col3.metric("Pas temporel", "30 min")
col4.metric("Meilleur modèle (prévision)", "Random Forest")

st.divider()

st.markdown("""
### Modules

| Page | Description |
|------|-------------|
| 📊 Classification & Clustering | KMeans pour détecter RP/RS, puis classification supervisée |
| 📈 Prévision | Prévision de la consommation sur une semaine |
| 🔄 Génération | Génération de courbes synthétiques (GAN + approche statistique) |
| 📉 Exploration | Visualisation exploratoire des profils de consommation |
""")

st.info("👈 Navigue entre les modules via le menu à gauche.")
