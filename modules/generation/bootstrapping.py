# Génération d'Indicateurs pour Analyser les Différences (Bootstrapping)
# Pour une comparaison quantitative des courbes générées par bootstrapping avec les courbes réelles, nous allons calculer les mêmes indicateurs statistiques clés que précédemment.

# --- Préparation des données réelles classifiées (si non déjà fait) ---
df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

rp_real_data = df_daily_classified[df_daily_classified['prediction_binaire'] == 0].drop(columns=['prediction_binaire'])
rs_real_data = df_daily_classified[df_daily_classified['prediction_binaire'] == 1].drop(columns=['prediction_binaire'])

# --- Calcul des indicateurs pour les courbes bootstrapped ---
indicators_bootstrap = []

indicators_bootstrap.append(calculate_consumption_indicators(rp_real_data, name='Réel RP'))
indicators_bootstrap.append(calculate_consumption_indicators(rs_real_data, name='Réel RS'))
indicators_bootstrap.append(calculate_consumption_indicators(synthetic_rp_curves_bootstrap, name='Synthétique RP (Bootstrapping)'))
indicators_bootstrap.append(calculate_consumption_indicators(synthetic_rs_curves_bootstrap, name='Synthétique RS (Bootstrapping)'))

# Création du DataFrame de comparaison
comparison_df_bootstrap = pd.DataFrame(indicators_bootstrap).set_index('Type')
display(comparison_df_bootstrap)

### 14. Génération de Courbes de Consommation Synthétiques (Approche Bootstrapping)

import numpy as np
import pandas as pd

def generate_synthetic_curves_bootstrap(df_base, df_results, consumer_type, num_curves=5, random_seed=42, time_unit='D'):
    """
    Génère des courbes de consommation synthétiques pour un type de consommateur donné en utilisant le bootstrapping.

    Args:
        df_base (pd.DataFrame): DataFrame des consommations (journalières ou hebdomadaires) (id x dates).
        df_results (pd.DataFrame): DataFrame avec l'affectation 'prediction_binaire' pour chaque id.
        consumer_type (int): 0 pour Résidence Principale, 1 pour Résidence Secondaire.
        num_curves (int): Nombre de courbes synthétiques à générer.
        random_seed (int): Graine pour la reproductibilité.
        time_unit (str): Unité de temps ('D' pour daily, 'W' pour weekly) pour l'indexation des noms de courbes.

    Returns:
        pd.DataFrame: DataFrame contenant les courbes synthétiques générées par bootstrapping.
    """
    np.random.seed(random_seed)

    # Joindre df_base avec les prédictions de type de consommateur
    df_classified = df_base.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

    # Filtrer les données pour le type de consommateur spécifié
    df_filtered = df_classified[df_classified['prediction_binaire'] == consumer_type].drop(columns=['prediction_binaire'])

    if df_filtered.empty:
        print(f"Aucune donnée réelle pour le type de consommateur {consumer_type} pour le bootstrapping.")
        return pd.DataFrame()

    # Sélectionner 'num_curves' courbes au hasard avec remplacement (bootstrapping)
    synthetic_df = df_filtered.sample(n=num_curves, replace=True, random_state=random_seed)
    synthetic_df.index = [f'synthetic_id_bootstrap_{consumer_type}_{time_unit}_{i}' for i in range(num_curves)]

    return synthetic_df

# --- Génération des courbes synthétiques journalières par bootstrapping ---
print("\n--- Génération de courbes journalières RP synthétiques par bootstrapping ---")
synthetic_rp_curves_bootstrap = generate_synthetic_curves_bootstrap(df_daily, df_results, consumer_type=0, num_curves=10, time_unit='D')
if not synthetic_rp_curves_bootstrap.empty:
    print(f"Généré {len(synthetic_rp_curves_bootstrap)} courbes RP synthétiques (bootstrapping).")
    display(synthetic_rp_curves_bootstrap.head())

print("\n--- Génération de courbes journalières RS synthétiques par bootstrapping ---")
synthetic_rs_curves_bootstrap = generate_synthetic_curves_bootstrap(df_daily, df_results, consumer_type=1, num_curves=10, time_unit='D')
if not synthetic_rs_curves_bootstrap.empty:
    print(f"Généré {len(synthetic_rs_curves_bootstrap)} courbes RS synthétiques (bootstrapping).")
    display(synthetic_rs_curves_bootstrap.head())

# Vérification de la Similarité et Cohérence Globale des Courbes Générées (Bootstrapping)
# Nous allons maintenant visualiser un échantillon de courbes réelles et synthétiques (générées par bootstrapping) pour chaque type de consommateur. Cela permettra d'apprécier visuellement la cohérence globale des données générées par cette nouvelle approche.

# Joindre df_daily avec les prédictions de type de consommateur
df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

# Sélectionner quelques courbes réelles pour chaque type
num_real_to_plot_bootstrap = 3

fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot pour les Résidences Principales (RP) - Bootstrapping
if not synthetic_rp_curves_bootstrap.empty:
    rp_real_sample_bootstrap = df_daily_classified[df_daily_classified['prediction_binaire'] == 0].drop(columns=['prediction_binaire']).sample(min(num_real_to_plot_bootstrap, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 0])), random_state=42)
    rp_synth_sample_bootstrap = synthetic_rp_curves_bootstrap.sample(min(num_real_to_plot_bootstrap, len(synthetic_rp_curves_bootstrap)), random_state=42)

    for i, row in rp_real_sample_bootstrap.iterrows():
        axes[0].plot(row.index, row.values, label=f'Réel RP {i}', alpha=0.7)
    for i, row in rp_synth_sample_bootstrap.iterrows():
        axes[0].plot(row.index, row.values, label=f'Synthétique RP (Boot.) {row.name}', linestyle='--', alpha=0.7)

    axes[0].set_title('Courbes de Consommation - Résidences Principales (Réelles vs Synthétiques Bootstrapping)')
    axes[0].set_ylabel('Consommation (Valeur)')
    axes[0].legend()
    axes[0].grid(True)

# Plot pour les Résidences Secondaires (RS) - Bootstrapping
if not synthetic_rs_curves_bootstrap.empty:
    rs_real_sample_bootstrap = df_daily_classified[df_daily_classified['prediction_binaire'] == 1].drop(columns=['prediction_binaire']).sample(min(num_real_to_plot_bootstrap, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 1])), random_state=42)
    rs_synth_sample_bootstrap = synthetic_rs_curves_bootstrap.sample(min(num_real_to_plot_bootstrap, len(synthetic_rs_curves_bootstrap)), random_state=42)

    for i, row in rs_real_sample_bootstrap.iterrows():
        axes[1].plot(row.index, row.values, label=f'Réel RS {i}', alpha=0.7)
    for i, row in rs_synth_sample_bootstrap.iterrows():
        axes[1].plot(row.index, row.values, label=f'Synthétique RS (Boot.) {row.name}', linestyle='--', alpha=0.7)

    axes[1].set_title('Courbes de Consommation - Résidences Secondaires (Réelles vs Synthétiques Bootstrapping)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Consommation (Valeur)')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# Vérification avec Moins de Courbes pour une Comparaison Plus Détaillée (Bootstrapping)
# Pour une analyse plus ciblée avec l'approche de bootstrapping, nous allons visualiser un seul exemple de courbe réelle et synthétique pour chaque type de consommateur.

# Joindre df_daily avec les prédictions de type de consommateur (si ce n'est pas déjà fait)
df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

# Sélectionner seulement 1 courbe réelle et 1 synthétique pour chaque type (Bootstrapping)
num_to_plot_bootstrap = 1

fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot pour les Résidences Principales (RP) - Bootstrapping
if not synthetic_rp_curves_bootstrap.empty:
    rp_real_sample_single_bootstrap = df_daily_classified[df_daily_classified['prediction_binaire'] == 0].drop(columns=['prediction_binaire']).sample(min(num_to_plot_bootstrap, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 0])), random_state=42)
    rp_synth_sample_single_bootstrap = synthetic_rp_curves_bootstrap.sample(min(num_to_plot_bootstrap, len(synthetic_rp_curves_bootstrap)), random_state=42)

    for i, row in rp_real_sample_single_bootstrap.iterrows():
        axes[0].plot(row.index, row.values, label=f'Réel RP {i}', alpha=0.7, color='blue')
    for i, row in rp_synth_sample_single_bootstrap.iterrows():
        axes[0].plot(row.index, row.values, label=f'Synthétique RP (Boot.) {row.name}', linestyle='--', alpha=0.7, color='orange')

    axes[0].set_title('Courbes de Consommation - Résidences Principales (Réelle vs Synthétique Bootstrapping - Exemple Unique)')
    axes[0].set_ylabel('Consommation (Valeur)')
    axes[0].legend()
    axes[0].grid(True)

# Plot pour les Résidences Secondaires (RS) - Bootstrapping
if not synthetic_rs_curves_bootstrap.empty:
    rs_real_sample_single_bootstrap = df_daily_classified[df_daily_classified['prediction_binaire'] == 1].drop(columns=['prediction_binaire']).sample(min(num_to_plot_bootstrap, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 1])), random_state=42)
    rs_synth_sample_single_bootstrap = synthetic_rs_curves_bootstrap.sample(min(num_to_plot_bootstrap, len(synthetic_rs_curves_bootstrap)), random_state=42)

    for i, row in rs_real_sample_single_bootstrap.iterrows():
        axes[1].plot(row.index, row.values, label=f'Réel RS {i}', alpha=0.7, color='blue')
    for i, row in rs_synth_sample_single_bootstrap.iterrows():
        axes[1].plot(row.index, row.values, label=f'Synthétique RS (Boot.) {row.name}', linestyle='--', alpha=0.7, color='orange')

    axes[1].set_title('Courbes de Consommation - Résidences Secondaires (Réelle vs Synthétique Bootstrapping - Exemple Unique)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Consommation (Valeur)')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.show()

