# %%
# =============================================================================
# BUILD FEATURES — Chargement CSV + Feature Engineering + Cache
# =============================================================================
# Ce script est à exécuter UNE SEULE FOIS (ou après mise à jour des données).
# Il sauvegarde le résultat dans un fichier cache .pkl pour que
# neuralnetwork&RF.py puisse le charger instantanément.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

# %%
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd() / "modules" / "classification"

raw_dataset = script_dir / ".." / ".." / "data" / "RES2-6-9.csv"
cache_path  = script_dir / ".." / ".." / "data" / "features_cache.pkl"

# =============================================================================
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES BRUTES
# =============================================================================
# %%
COL_PDL = "pdl_id"
COL_DT  = "datetime"
COL_PWR = "p_kw"

print("⏳ Chargement du CSV (opération longue)...")
raw = pd.read_csv(raw_dataset, sep=',', decimal='.')
raw = raw.rename(columns={"ID": COL_PDL, "id": COL_PDL,
                           "horodate": COL_DT, "valeur": COL_PWR})

raw[COL_DT] = pd.to_datetime(raw[COL_DT], utc=True, errors="coerce")

if raw[COL_DT].dt.tz is None:
    raw[COL_DT] = raw[COL_DT].dt.tz_localize("Europe/Paris")
else:
    raw[COL_DT] = raw[COL_DT].dt.tz_convert("Europe/Paris")

df = raw.dropna(subset=[COL_PDL, COL_DT, COL_PWR]).copy()
df[COL_PWR] = pd.to_numeric(df[COL_PWR], errors="coerce")
df = df.dropna(subset=[COL_PWR])

print(f"✅ Données chargées : {len(df)} lignes, {df[COL_PDL].nunique()} clients")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

# %%
# --- Colonnes temporelles ---
df["date"]       = df[COL_DT].dt.date
df["dow"]        = df[COL_DT].dt.dayofweek
df["is_weekend"] = df["dow"] >= 5
df["hh_index"]   = ((df[COL_DT].dt.hour * 60) + df[COL_DT].dt.minute) // 30

# %%
# --- Agrégation journalière par client ---
daily = (df
    .assign(energy_kwh_step=df[COL_PWR] * 0.5)
    .groupby([COL_PDL, "date"], as_index=False)
    .agg(
        daily_kwh=("energy_kwh_step", "sum"),
        daily_mean_kw=(COL_PWR, "mean"),
        daily_max_kw=(COL_PWR, "max"),
        n_steps=(COL_PWR, "size"),
    )
)
print(f"daily: {daily.shape}")

# %%
# --- Seuil d'activité par client (Q20 des jours > 0) ---
def q20_positive(s: pd.Series):
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    return s.quantile(0.2)

daily["th_pdl"] = (daily.groupby(COL_PDL)["daily_kwh"]
                   .transform(q20_positive))
daily["is_active_day"] = (daily["daily_kwh"] >= daily["th_pdl"]).fillna(False)

# %%
# --- Saisonnalité ---
daily2 = daily.copy()
daily2["date_ts"] = pd.to_datetime(daily2["date"])
daily2["month"]   = daily2["date_ts"].dt.month

def season_from_month(m):
    if m in (12, 1, 2):
        return "winter"
    if m in (6, 7, 8):
        return "summer"
    return "mid"

daily2["season"] = daily2["month"].map(season_from_month)

season_stats = (daily2
    .groupby([COL_PDL, "season"], as_index=False)
    .agg(mean_daily_kwh=("daily_kwh", "mean"))
    .pivot(index=COL_PDL, columns="season", values="mean_daily_kwh")
    .reset_index()
)
for c in ["winter", "summer", "mid"]:
    if c not in season_stats.columns:
        season_stats[c] = 0.0

global_mean = (daily2.groupby(COL_PDL, as_index=False)
               .agg(mean_daily_kwh_global=("daily_kwh", "mean")))

season_stats = season_stats.merge(global_mean, on=COL_PDL, how="left", validate="one_to_one")

eps = 1e-9
season_stats["r_global"] = 1.0
season_stats["r_mid"]    = season_stats["mid"]    / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_summer"] = season_stats["summer"] / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_winter"] = season_stats["winter"] / (season_stats["mean_daily_kwh_global"] + eps)

season_stats = season_stats[[COL_PDL, "r_global", "r_mid", "r_summer", "r_winter"]]

# %%
# --- Activité globale par client ---
activity = (daily
    .groupby(COL_PDL, as_index=False)
    .agg(
        n_days=(COL_PDL, "size"),
        n_active_days=("is_active_day", "sum"),
        active_day_rate=("is_active_day", "mean"),
        mean_daily_kwh=("daily_kwh", "mean"),
        p95_daily_kwh=("daily_kwh", lambda s: s.quantile(0.95)),
        cv_daily_kwh=("daily_kwh", lambda s: (s.std() / s.mean()) if s.mean() != 0 else np.nan),
    )
)

# %%
# --- Runs & Gaps (séquences de jours actifs/inactifs) ---
def runs_and_gaps(active_series: pd.Series):
    runs = []
    gaps = []
    run = 0
    gap = 0
    for v in active_series.astype(bool):
        if v:
            run += 1
            if gap > 0:
                gaps.append(gap)
                gap = 0
        else:
            gap += 1
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    if gap > 0:
        gaps.append(gap)
    return pd.Series({
        "n_runs": len(runs),
        "mean_run_len": float(np.mean(runs)) if runs else 0.0,
        "max_run_len": float(np.max(runs)) if runs else 0.0,
        "mean_gap_len": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_len": float(np.max(gaps)) if gaps else 0.0,
    })

runs_stats = (daily
    .sort_values([COL_PDL, "date"])
    .groupby(COL_PDL)["is_active_day"]
    .apply(runs_and_gaps)
    .unstack()
    .reset_index()
)

# %%
# --- Pattern semaine/week-end ---
daily_dt = pd.to_datetime(daily["date"])
daily["dow"]        = daily_dt.dt.dayofweek
daily["is_weekend"] = daily["dow"] >= 5

week_pattern = (daily
    .groupby([COL_PDL, "is_weekend"], as_index=False)
    .agg(active_rate=("is_active_day", "mean"),
         mean_kwh=("daily_kwh", "mean"))
    .pivot(index=COL_PDL, columns="is_weekend")
)
week_pattern.columns = [f"{a}_{'weekend' if b else 'weekday'}" for a, b in week_pattern.columns]
week_pattern = week_pattern.reset_index()

# %%
# --- Fusion de toutes les features ---
features_pdl = (activity
    .merge(runs_stats,   on=COL_PDL, how="left", validate="one_to_one")
    .merge(week_pattern, on=COL_PDL, how="left", validate="one_to_one")
    .merge(season_stats, on=COL_PDL, how="left", validate="one_to_one")
)

features_pdl["seasonality_amp"]     = (features_pdl[["r_mid","r_summer","r_winter"]].max(axis=1)
                                     - features_pdl[["r_mid","r_summer","r_winter"]].min(axis=1))
features_pdl["winter_minus_summer"] = features_pdl["r_winter"] - features_pdl["r_summer"]

assert features_pdl[COL_PDL].is_unique, "Doublons détectés après merge !"
print(f"\n✅ Feature engineering terminé : {len(features_pdl)} clients, {features_pdl.shape[1]} colonnes")

# =============================================================================
# 3. SAUVEGARDE DU CACHE
# =============================================================================
# %%
features_pdl.to_pickle(cache_path)
print(f"\n💾 Cache sauvegardé dans : {cache_path}")
print("   → Relancez maintenant neuralnetwork&RF.py (chargement instantané)")
