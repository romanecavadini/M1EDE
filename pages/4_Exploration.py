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
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("📉 Exploration & Visualisation des courbes de consommation")
st.markdown("Analyse visuelle des profils de consommation par type de résidence — Salma")

DATA_PATH   = Path("data/export.csv")
LABELS_PATH = Path("data/RES2-6-9-labels.csv")

# ── Chargement et préparation ──
@st.cache_data
def load_and_prepare():
    # Données de consommation
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"id": "pdl_id", "horodate": "datetime", "valeur": "p_kw"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["pdl_id", "datetime", "p_kw"])

    # Pivot : index = pdl_id, colonnes = datetime
    df_daily = df.pivot_table(
        index="pdl_id", columns="datetime", values="p_kw", aggfunc="first"
    ).fillna(0)

    # Labels
    labels = pd.read_csv(LABELS_PATH, sep=';')
    labels['id'] = labels['id'].astype(str)
    df_daily.index = df_daily.index.astype(str)

    # df_results avec consumer_type
    df_results = labels.set_index('id')[['label', 'cluster']].copy()
    df_results['prediction_binaire'] = df_results['label']
    df_results['consumer_type'] = df_results['prediction_binaire'].map({
        0: 'Principale', 1: 'Secondaire'
    })

    return df_daily, df_results

with st.spinner("Chargement des données..."):
    df_daily, df_results = load_and_prepare()

clients = df_daily.index.tolist()

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Courbes par type",
    "👤 Client vs moyenne",
    "📅 Analyse hebdomadaire",
    "📦 Box plots"
])

# =============================================================================
# TAB 1 — Courbes agrégées par type
# =============================================================================
with tab1:
    st.subheader("Courbes de consommation agrégées par type de résidence")

    col1, col2, col3 = st.columns(3)
    with col1:
        resample_freq = st.selectbox("Granularité", ["D", "W", "ME"], 
                                      format_func=lambda x: {"D": "Journalier", "W": "Hebdomadaire", "ME": "Mensuel"}[x])
    with col2:
        agg_method = st.selectbox("Agrégation", ["sum", "mean", "median"],
                                   format_func=lambda x: {"sum": "Somme", "mean": "Moyenne", "median": "Médiane"}[x])
    with col3:
        show_error = st.checkbox("Afficher bandes d'écart-type", value=False)

    if st.button("Générer les courbes par type"):
        fig = go.Figure()
        colors = {"Principale": "royalblue", "Secondaire": "tomato"}

        for type_res in ["Principale", "Secondaire"]:
            ids = df_results[df_results["consumer_type"] == type_res].index.tolist()
            ids = [i for i in ids if i in df_daily.index]
            if not ids:
                continue

            sub = df_daily.loc[ids].copy()
            sub.columns = pd.to_datetime(sub.columns)
            series = sub.stack().rename("value")
            series.index = series.index.map(lambda x: x[1])
            series.index = pd.to_datetime(series.index)

            resampled = getattr(series.resample(resample_freq), agg_method)()

            fig.add_trace(go.Scatter(
                x=resampled.index, y=resampled.values,
                name=type_res, line=dict(color=colors[type_res], width=2)
            ))

            if show_error:
                std = series.resample(resample_freq).std()
                fig.add_trace(go.Scatter(
                    x=resampled.index.tolist() + resampled.index.tolist()[::-1],
                    y=(resampled + std).tolist() + (resampled - std).tolist()[::-1],
                    fill='toself', fillcolor=colors[type_res],
                    opacity=0.15, line=dict(color='rgba(255,255,255,0)'),
                    name=f"{type_res} ±1 std", showlegend=True
                ))

        fig.update_layout(
            title="Consommation par type de résidence",
            xaxis_title="Date", yaxis_title="Consommation (kWh)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — Client vs moyenne de son type
# =============================================================================
with tab2:
    st.subheader("Consommation d'un client vs moyenne de son type")

    client_id = st.selectbox("Sélectionne un client", clients, key="tab2_client")
    freq2 = st.selectbox("Granularité", ["D", "W", "ME"],
                          format_func=lambda x: {"D": "Journalier", "W": "Hebdomadaire", "ME": "Mensuel"}[x],
                          key="tab2_freq")

    if st.button("Comparer", key="btn_tab2"):
        if client_id not in df_results.index:
            st.error("Client non trouvé dans les labels.")
        else:
            consumer_type = df_results.loc[client_id, "consumer_type"]
            st.info(f"Ce client est une **{consumer_type}**")

            # Courbe client
            client_series = df_daily.loc[client_id].copy()
            client_series.index = pd.to_datetime(client_series.index)
            client_resampled = getattr(client_series.resample(freq2), "sum")()

            # Moyenne du type
            type_ids = [i for i in df_results[df_results["consumer_type"] == consumer_type].index if i in df_daily.index and i != client_id]
            type_sub = df_daily.loc[type_ids].copy()
            type_sub.columns = pd.to_datetime(type_sub.columns)
            type_series = type_sub.stack().rename("value")
            type_series.index = type_series.index.map(lambda x: x[1])
            type_series.index = pd.to_datetime(type_series.index)
            type_avg = type_series.resample(freq2).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=client_resampled.index, y=client_resampled.values,
                name=f"Client {client_id}", line=dict(color="royalblue", width=2)
            ))
            fig2.add_trace(go.Scatter(
                x=type_avg.index, y=type_avg.values,
                name=f"Moyenne {consumer_type}",
                line=dict(color="tomato", width=2, dash="dash")
            ))
            fig2.update_layout(
                title=f"Client {client_id} vs moyenne {consumer_type}",
                xaxis_title="Date", yaxis_title="Consommation (kWh)",
                hovermode="x unified"
            )
            st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# TAB 3 — Analyse hebdomadaire
# =============================================================================
with tab3:
    st.subheader("Analyse sur une semaine spécifique")

    client_id3 = st.selectbox("Client", clients, key="tab3_client")
    start_date = st.date_input("Semaine commençant le", key="tab3_date")
    freq3 = st.selectbox("Granularité", ["h", "D"],
                          format_func=lambda x: {"h": "Horaire", "D": "Journalier"}[x],
                          key="tab3_freq")

    if st.button("Analyser la semaine", key="btn_tab3"):
        if client_id3 not in df_results.index:
            st.error("Client non trouvé.")
        else:
            consumer_type3 = df_results.loc[client_id3, "consumer_type"]
            start = pd.to_datetime(start_date, utc=True)
            end   = start + pd.Timedelta(weeks=1)

            # Filtre sur la semaine
            cols_week = df_daily.columns[
                (pd.to_datetime(df_daily.columns) >= start) &
                (pd.to_datetime(df_daily.columns) < end)
            ]

            if len(cols_week) == 0:
                st.error("Aucune donnée pour cette semaine.")
            else:
                weekly = df_daily[cols_week]

                client_week = weekly.loc[client_id3].copy()
                client_week.index = pd.to_datetime(client_week.index)
                client_week_resampled = getattr(client_week.resample(freq3), "sum")()

                type_ids3 = [i for i in df_results[df_results["consumer_type"] == consumer_type3].index if i in df_daily.index and i != client_id3]
                type_week = weekly.loc[type_ids3].stack().rename("value")
                type_week.index = type_week.index.map(lambda x: x[1])
                type_week.index = pd.to_datetime(type_week.index)
                type_week_avg = type_week.resample(freq3).mean()

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=client_week_resampled.index, y=client_week_resampled.values,
                    name=f"Client {client_id3}", line=dict(color="royalblue", width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=type_week_avg.index, y=type_week_avg.values,
                    name=f"Moyenne {consumer_type3}",
                    line=dict(color="tomato", width=2, dash="dash")
                ))
                fig3.update_layout(
                    title=f"Semaine du {start_date} — Client vs moyenne {consumer_type3}",
                    xaxis_title="Date", yaxis_title="Consommation (kWh)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# TAB 4 — Box plots
# =============================================================================
with tab4:
    st.subheader("Distribution de la consommation par heure et jour de la semaine")

    if st.button("Générer les box plots", key="btn_tab4"):
        with st.spinner("Préparation des données..."):
            df_flat = df_daily.reset_index()
            df_long = df_flat.melt(id_vars='pdl_id', var_name='datetime', value_name='consumption')
            df_long['datetime'] = pd.to_datetime(df_long['datetime'])
            df_long['hour']     = df_long['datetime'].dt.hour
            df_long['day']      = df_long['datetime'].dt.day_name()
            df_long['pdl_id']   = df_long['pdl_id'].astype(str)
            df_long = df_long.merge(
                df_results[['consumer_type']],
                left_on='pdl_id', right_index=True, how='left'
            )

        # Box plot par heure
        fig4 = px.box(
            df_long, x='hour', y='consumption', color='consumer_type',
            title="Distribution de la consommation par heure",
            labels={"hour": "Heure", "consumption": "Consommation (kWh)", "consumer_type": "Type"},
            color_discrete_map={"Principale": "royalblue", "Secondaire": "tomato"}
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Box plot par jour
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        fig5 = px.box(
            df_long, x='day', y='consumption', color='consumer_type',
            category_orders={"day": day_order},
            title="Distribution de la consommation par jour de la semaine",
            labels={"day": "Jour", "consumption": "Consommation (kWh)", "consumer_type": "Type"},
            color_discrete_map={"Principale": "royalblue", "Secondaire": "tomato"}
        )
        st.plotly_chart(fig5, use_container_width=True)
