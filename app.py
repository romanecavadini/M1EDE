import streamlit as st
import subprocess

subprocess.run(["python", "download_data.py"])

st.set_page_config(
    page_title="Data & Énergie",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard Data & Énergie")
st.markdown("Analyse de la consommation électrique des particuliers — M1 EDE")
st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Foyers analysés", "500")
col2.metric("Période", "2023 – 2024")
col3.metric("Modules", "3")

st.info("👈 Navigue entre les modules via le menu à gauche.")
