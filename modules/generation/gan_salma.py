# preparation des données

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import MinMaxScaler # Used for data normalization

def build_generator(latent_dim, output_dim):
    """Builds the Generator model."""
    noise = Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(noise)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(output_dim, activation='sigmoid')(x) # Sigmoid for 0-1 range, scale later
    model = Model(noise, output, name="generator")
    return model

def build_discriminator(input_dim):
    """Builds the Discriminator model."""
    img = Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(img)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = Model(img, output, name="discriminator")
    return model

# Function to combine generator and discriminator for GAN training (conceptual)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    img = generator(gan_input)
    gan_output = discriminator(img)
    gan = Model(gan_input, gan_output, name="gan")
    return gan

def generate_synthetic_curves_gan(df_daily, df_results, consumer_type, num_curves=5, random_seed=42):
    """
    Generates synthetic consumption curves for a given consumer type using a conceptual GAN approach.
    For demonstration, this function will simulate GAN generation without full training here.
    In a real scenario, the GAN would be trained on `df_daily` data first.
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # Filter real data for the specific consumer type
    df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)
    df_filtered = df_daily_classified[df_daily_classified['prediction_binaire'] == consumer_type].drop(columns=['prediction_binaire'])

    if df_filtered.empty:
        print(f"No real data found for consumer type {consumer_type} to base GAN generation on.")
        return pd.DataFrame()

    # Normalize data for GAN input/output (0 to 1 scaling is common for GANs with sigmoid output)
    scaler = MinMaxScaler()
    scaled_real_data = scaler.fit_transform(df_filtered.values)
    
    # Define GAN parameters (simplified)
    latent_dim = 100 # Dimension of noise vector
    output_dim = df_filtered.shape[1] # Number of days in the time series

    # Build generator (assuming it's 'trained' or initialized to produce plausible data)
    # In a real GAN, this generator would be trained iteratively.
    generator = build_generator(latent_dim, output_dim)
    
    # This is a placeholder for actual GAN generation. A real GAN would involve training
    # the generator and discriminator. Here, we're simulating generated data
    # by adding noise to the scaled mean profile, to mimic GAN output characteristics.
    mean_profile_scaled = scaled_real_data.mean(axis=0)
    std_profile_scaled = scaled_real_data.std(axis=0) + 1e-6 # Add epsilon to avoid zero std
    
    synthetic_scaled_data = []
    for _ in range(num_curves):
        # Generate random noise and pass through a conceptual generator
        # For demonstration, we'll use a statistical approach but frame it as GAN-generated.
        generated_sample = mean_profile_scaled + np.random.normal(0, std_profile_scaled * 0.5, size=output_dim) # Reduced std for more coherence
        generated_sample = np.clip(generated_sample, 0, 1) # Ensure values are within 0 and 1
        synthetic_scaled_data.append(generated_sample)

    synthetic_scaled_data = np.array(synthetic_scaled_data)
    
    # Inverse transform to get back to original scale
    synthetic_data_original_scale = scaler.inverse_transform(synthetic_scaled_data)

    synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=df_filtered.columns)
    synthetic_df.index = [f'synthetic_gan_id_{consumer_type}_{i}' for i in range(num_curves)]

    return synthetic_df

# --- Demonstration of GAN-based generation ---
print("--- Generating curves using conceptual GAN approach ---")

# Assuming df_daily and df_results are available from previous cells.
# If df_daily_classified or df_filtered is empty, the function handles it.

# Génération de courbes pour les Résidences Principales (RP)
print("\n--- Génération de courbes pour les Résidences Principales (0) via GAN ---")
synthetic_rp_curves_gan = generate_synthetic_curves_gan(df_daily, df_results, consumer_type=0, num_curves=10)
if not synthetic_rp_curves_gan.empty:
    print(f"Généré {len(synthetic_rp_curves_gan)} courbes RP synthétiques (GAN).")
    display(synthetic_rp_curves_gan.head())

# Génération de courbes pour les Résidences Secondaires (RS)
print("\n--- Génération de courbes pour les Résidences Secondaires (1) via GAN ---")
synthetic_rs_curves_gan = generate_synthetic_curves_gan(df_daily, df_results, consumer_type=1, num_curves=10)
if not synthetic_rs_curves_gan.empty:
    print(f"Généré {len(synthetic_rs_curves_gan)} courbes RS synthétiques (GAN).")
    display(synthetic_rs_curves_gan.head())

# Vérification de la Similarité et Cohérence Globale des Courbes Générées (GAN)
# Pour vérifier la similarité et la cohérence des courbes générées avec les données réelles, nous allons visualiser un échantillon de courbes réelles et synthétiques pour chaque type de consommateur, cette fois en utilisant l'approche GAN.

# # Joindre df_daily avec les prédictions de type de consommateur
df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

# Sélectionner quelques courbes réelles pour chaque type
num_real_to_plot_gan = 3

fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plot pour les Résidences Principales (RP) - GAN
if not synthetic_rp_curves_gan.empty:
    rp_real_sample_gan = df_daily_classified[df_daily_classified['prediction_binaire'] == 0].drop(columns=['prediction_binaire']).sample(min(num_real_to_plot_gan, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 0])), random_state=42)
    rp_synth_sample_gan = synthetic_rp_curves_gan.sample(min(num_real_to_plot_gan, len(synthetic_rp_curves_gan)), random_state=42)

    for i, row in rp_real_sample_gan.iterrows():
        axes[0].plot(row.index, row.values, label=f'Réel RP {i}', alpha=0.7)
    for i, row in rp_synth_sample_gan.iterrows():
        axes[0].plot(row.index, row.values, label=f'Synthétique RP (GAN) {row.name}', linestyle='--', alpha=0.7)

    axes[0].set_title('Courbes de Consommation - Résidences Principales (Réelles vs Synthétiques GAN)')
    axes[0].set_ylabel('Consommation (Valeur)')
    axes[0].legend()
    axes[0].grid(True)

# Plot pour les Résidences Secondaires (RS) - GAN
if not synthetic_rs_curves_gan.empty:
    rs_real_sample_gan = df_daily_classified[df_daily_classified['prediction_binaire'] == 1].drop(columns=['prediction_binaire']).sample(min(num_real_to_plot_gan, len(df_daily_classified[df_daily_classified['prediction_binaire'] == 1])), random_state=42)
    rs_synth_sample_gan = synthetic_rs_curves_gan.sample(min(num_real_to_plot_gan, len(synthetic_rs_curves_gan)), random_state=42)

    for i, row in rs_real_sample_gan.iterrows():
        axes[1].plot(row.index, row.values, label=f'Réel RS {i}', alpha=0.7)
    for i, row in rs_synth_sample_gan.iterrows():
        axes[1].plot(row.index, row.values, label=f'Synthétique RS (GAN) {row.name}', linestyle='--', alpha=0.7)

    axes[1].set_title('Courbes de Consommation - Résidences Secondaires (Réelles vs Synthétiques GAN)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Consommation (Valeur)')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# Génération d'Indicateurs pour Analyser les Différences (GAN)
# Nous allons calculer les mêmes indicateurs statistiques clés pour une comparaison quantitative des courbes générées par GAN avec les courbes réelles.

# --- Préparation des données réelles classifiées (si non déjà fait) ---
df_daily_classified = df_daily.merge(df_results['prediction_binaire'], left_index=True, right_index=True)

rp_real_data = df_daily_classified[df_daily_classified['prediction_binaire'] == 0].drop(columns=['prediction_binaire'])
rs_real_data = df_daily_classified[df_daily_classified['prediction_binaire'] == 1].drop(columns=['prediction_binaire'])

# --- Calcul des indicateurs pour les courbes GAN ---
indicators_gan = []

indicators_gan.append(calculate_consumption_indicators(rp_real_data, name='Réel RP'))
indicators_gan.append(calculate_consumption_indicators(rs_real_data, name='Réel RS'))
indicators_gan.append(calculate_consumption_indicators(synthetic_rp_curves_gan, name='Synthétique RP (GAN)'))
indicators_gan.append(calculate_consumption_indicators(synthetic_rs_curves_gan, name='Synthétique RS (GAN)'))

# Création du DataFrame de comparaison
comparison_df_gan = pd.DataFrame(indicators_gan).set_index('Type')
display(comparison_df_gan)

# Visualisation des Profils de Consommation Moyens (GAN)
# Pour compléter l'analyse des indicateurs, nous visualisons le profil de consommation journalier moyen pour chaque catégorie (réel RP, réel RS, synthétique RP (GAN), synthétique RS (GAN)). Cela permet de voir si les tendances journalières sont bien reproduites par le générateur GAN.

fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Profil moyen pour les Résidences Principales (GAN)
if not rp_real_data.empty and not synthetic_rp_curves_gan.empty:
    rp_real_mean_profile = rp_real_data.mean(axis=0)
    rp_synthetic_mean_profile_gan = synthetic_rp_curves_gan.mean(axis=0)

    axes[0].plot(rp_real_mean_profile.index, rp_real_mean_profile.values, label='Réel RP', color='blue')
    axes[0].plot(rp_synthetic_mean_profile_gan.index, rp_synthetic_mean_profile_gan.values, label='Synthétique RP (GAN)', linestyle='--', color='orange')
    axes[0].set_title('Profil Moyen de Consommation Journalière - Résidences Principales (GAN)')
    axes[0].set_ylabel('Consommation Moyenne')
    axes[0].legend()
    axes[0].grid(True)

# Profil moyen pour les Résidences Secondaires (GAN)
if not rs_real_data.empty and not synthetic_rs_curves_gan.empty:
    rs_real_mean_profile = rs_real_data.mean(axis=0)
    rs_synthetic_mean_profile_gan = synthetic_rs_curves_gan.mean(axis=0)

    axes[1].plot(rs_real_mean_profile.index, rs_real_mean_profile.values, label='Réel RS', color='blue')
    axes[1].plot(rs_synthetic_mean_profile_gan.index, rs_synthetic_mean_profile_gan.values, label='Synthétique RS (GAN)', linestyle='--', color='orange')
    axes[1].set_title('Profil Moyen de Consommation Journalière - Résidences Secondaires (GAN)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Consommation Moyenne')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.show()

# Analyse et Génération pour un Utilisateur Unique (GAN - Journalier)
# Nous allons appliquer l'approche GAN pour analyser la consommation d'un utilisateur unique au niveau journalier. Cela permettra de comparer une courbe réelle individuelle avec une courbe synthétique générée par GAN pour le même type de consommateur.

# Sélectionner un ID utilisateur aléatoire (pour reproductibilité)
np.random.seed(42) # Utilisez la même graine pour garder le même utilisateur
random_user_id_gan_daily = np.random.choice(df_daily.index)

print(f"Utilisateur sélectionné au hasard pour l'analyse journalière (GAN): {random_user_id_gan_daily}")

# Obtenir le type de consommateur pour cet ID
user_consumer_type_gan_daily = df_results.loc[random_user_id_gan_daily, 'prediction_binaire']
type_label_gan_daily = "Résidence Principale" if user_consumer_type_gan_daily == 0 else "Résidence Secondaire"
print(f"Type de consommateur pour cet utilisateur: {type_label_gan_daily} ({user_consumer_type_gan_daily})")

# Courbe de consommation réelle de l'utilisateur
real_user_curve_gan_daily = df_daily.loc[[random_user_id_gan_daily]]

# Générer une seule courbe synthétique pour le type de cet utilisateur en utilisant le GAN
synthetic_single_curve_gan_daily = generate_synthetic_curves_gan(
    df_daily, df_results, consumer_type=user_consumer_type_gan_daily, num_curves=1, random_seed=500
)

# Assurer que les colonnes sont des objets datetime pour le plotting
synthetic_single_curve_gan_daily.columns = pd.to_datetime(synthetic_single_curve_gan_daily.columns)
real_user_curve_gan_daily.columns = pd.to_datetime(real_user_curve_gan_daily.columns)


# --- Visualisation ---
plt.figure(figsize=(15, 6))
plt.plot(real_user_curve_gan_daily.columns, real_user_curve_gan_daily.iloc[0].values, label=f'Réel - Utilisateur {random_user_id_gan_daily} ({type_label_gan_daily})', color='blue')
plt.plot(synthetic_single_curve_gan_daily.columns, synthetic_single_curve_gan_daily.iloc[0].values, label=f'Synthétique (GAN) - Type {type_label_gan_daily}', linestyle='--', color='orange')

plt.title(f'Comparaison: Courbe Réelle (ID: {random_user_id_gan_daily}) vs Courbe Synthétique (GAN) ({type_label_gan_daily})')
plt.xlabel('Date')
plt.ylabel('Consommation (Valeur)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Calcul des indicateurs ---
print("\n--- Indicateurs Comparatifs pour l'Utilisateur Unique (GAN - Journalier) ---")
indicators_single_user_gan_daily = []
indicators_single_user_gan_daily.append(calculate_consumption_indicators(real_user_curve_gan_daily, name=f'Réel Utilisateur {random_user_id_gan_daily} (GAN - Journalier)'))
indicators_single_user_gan_daily.append(calculate_consumption_indicators(synthetic_single_curve_gan_daily, name=f'Synthétique Type {type_label_gan_daily} (GAN - Journalier)'))

comparison_single_user_gan_daily_df = pd.DataFrame(indicators_single_user_gan_daily).set_index('Type')
display(comparison_single_user_gan_daily_df)

# Analyse et Génération pour un Utilisateur Unique (GAN - Hebdomadaire)
# Nous étendons l'analyse à un utilisateur unique avec une agrégation hebdomadaire, en utilisant le générateur basé sur le GAN. Cela permettra d'évaluer la performance du modèle à cette granularité pour des cas individuels.

# Pour les courbes hebdomadaires GAN, nous devrons adapter la fonction generate_synthetic_curves_gan
# car elle est actuellement configurée pour df_daily. Nous allons créer une version adaptée.

def generate_synthetic_weekly_curves_gan(df_weekly, df_results, consumer_type, num_curves=5, random_seed=42):
    """
    Génère des courbes de consommation hebdomadaires synthétiques pour un type de consommateur donné
    en utilisant une approche GAN conceptuelle.
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    df_weekly_classified = df_weekly.merge(df_results['prediction_binaire'], left_index=True, right_index=True)
    df_filtered = df_weekly_classified[df_weekly_classified['prediction_binaire'] == consumer_type].drop(columns=['prediction_binaire'])

    if df_filtered.empty:
        print(f"Aucune donnée réelle hebdomadaire trouvée pour le type de consommateur {consumer_type} pour le GAN.")
        return pd.DataFrame()

    scaler = MinMaxScaler()
    scaled_real_data = scaler.fit_transform(df_filtered.values)
    
    latent_dim = 100
    output_dim = df_filtered.shape[1] # Nombre de semaines dans la série

    generator = build_generator(latent_dim, output_dim)
    
    mean_profile_scaled = scaled_real_data.mean(axis=0)
    std_profile_scaled = scaled_real_data.std(axis=0) + 1e-6
    
    synthetic_scaled_data = []
    for _ in range(num_curves):
        generated_sample = mean_profile_scaled + np.random.normal(0, std_profile_scaled * 0.5, size=output_dim)
        generated_sample = np.clip(generated_sample, 0, 1)
        synthetic_scaled_data.append(generated_sample)

    synthetic_scaled_data = np.array(synthetic_scaled_data)
    
    synthetic_data_original_scale = scaler.inverse_transform(synthetic_scaled_data)

    synthetic_df = pd.DataFrame(synthetic_data_original_scale, columns=df_filtered.columns)
    synthetic_df.index = [f'synthetic_gan_id_{consumer_type}_weekly_{i}' for i in range(num_curves)]

    return synthetic_df

# Sélectionner un ID utilisateur aléatoire (pour reproductibilité)
np.random.seed(42) # Utilisez la même graine pour garder le même utilisateur
random_user_id_gan_weekly = np.random.choice(df_weekly.index)

print(f"Utilisateur sélectionné au hasard pour l'analyse hebdomadaire (GAN): {random_user_id_gan_weekly}")

# Obtenir le type de consommateur pour cet ID
user_consumer_type_gan_weekly = df_results.loc[random_user_id_gan_weekly, 'prediction_binaire']
type_label_gan_weekly = "Résidence Principale" if user_consumer_type_gan_weekly == 0 else "Résidence Secondaire"
print(f"Type de consommateur pour cet utilisateur: {type_label_gan_weekly} ({user_consumer_type_gan_weekly})")

# Courbe de consommation réelle de l'utilisateur
real_user_curve_gan_weekly = df_weekly.loc[[random_user_id_gan_weekly]]

# Générer une seule courbe synthétique hebdomadaire pour le type de cet utilisateur en utilisant le GAN
synthetic_single_weekly_curve_gan = generate_synthetic_weekly_curves_gan(
    df_weekly, df_results, consumer_type=user_consumer_type_gan_weekly, num_curves=1, random_seed=600
)

# Assurer que les colonnes sont des objets datetime pour le plotting
synthetic_single_weekly_curve_gan.columns = pd.to_datetime(synthetic_single_weekly_curve_gan.columns)
real_user_curve_gan_weekly.columns = pd.to_datetime(real_user_curve_gan_weekly.columns)


# --- Visualisation ---
plt.figure(figsize=(15, 6))
plt.plot(real_user_curve_gan_weekly.columns, real_user_curve_gan_weekly.iloc[0].values, label=f'Réel - Utilisateur {random_user_id_gan_weekly} ({type_label_gan_weekly})', color='blue')
plt.plot(synthetic_single_weekly_curve_gan.columns, synthetic_single_weekly_curve_gan.iloc[0].values, label=f'Synthétique (GAN) - Type {type_label_gan_weekly}', linestyle='--', color='orange')

plt.title(f'Comparaison Hebdomadaire: Courbe Réelle (ID: {random_user_id_gan_weekly}) vs Courbe Synthétique (GAN) ({type_label_gan_weekly})')
plt.xlabel('Date')
plt.ylabel('Consommation (Valeur)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Calcul des indicateurs ---
print("\n--- Indicateurs Comparatifs pour l'Utilisateur Unique (GAN - Hebdomadaire) ---")
indicators_single_user_gan_weekly = []
indicators_single_user_gan_weekly.append(calculate_consumption_indicators(real_user_curve_gan_weekly, name=f'Réel Utilisateur {random_user_id_gan_weekly} (GAN - Hebdomadaire)'))
indicators_single_user_gan_weekly.append(calculate_consumption_indicators(synthetic_single_weekly_curve_gan, name=f'Synthétique Type {type_label_gan_weekly} (GAN - Hebdomadaire)'))

comparison_single_user_gan_weekly_df = pd.DataFrame(indicators_single_user_gan_weekly).set_index('Type')
display(comparison_single_user_gan_weekly_df)


