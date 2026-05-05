# %%
# =============================================================================
# GAN EXPERIMENT — Génération de courbes de charge (pas de 30 min)
# =============================================================================
# Ce script contient toute la pipeline : Chargement, Préparation des tenseurs,
# Définition des Modèles PyTorch, Entraînement des GANs séparés (RP/RS),
# et visualisation des résultats.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Définition de l'appareil (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Chemins
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd() / "M1EDE" / "Code" / "Generation"

data_dir = script_dir / ".." / ".." / "data"
raw_dataset = data_dir / "RES2-6-9.csv"
labels_path = data_dir / "RES2-6-9-labels.csv"

# Hyperparamètres temporels
STEPS_PER_HOUR = 2
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
SEQ_LEN = STEPS_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK  # 336

# %%
# =============================================================================
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES BRUTES
# =============================================================================
COL_PDL = "pdl_id"
COL_DT  = "datetime"
COL_PWR = "p_kw"

print("⏳ Chargement du CSV de consommation...")
raw = pd.read_csv(raw_dataset, sep=';', decimal='.')
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

print("⏳ Chargement des labels...")
df_labels = pd.read_csv(labels_path, sep=',')
df_labels = df_labels.rename(columns={"id": COL_PDL})

# On fusionne pour avoir le type (0 = RP, 1 = RS a priori)
df = df.merge(df_labels[[COL_PDL, 'label']], on=COL_PDL, how='inner')

# %%
# =============================================================================
# 2. EXTRACTION DES SÉQUENCES HEBDOMADAIRES (336 pas)
# =============================================================================
print("⏳ Extraction des semaines complètes (séquences de taille 336)...")

# Pour isoler les semaines, on utilise l'année et la semaine ISO
df['year_week'] = df[COL_DT].dt.strftime('%G-%V')
df = df.sort_values(by=[COL_PDL, COL_DT])

# On groupe par (pdl_id, year_week) et on vérifie si on a exactement 336 pas
sequences = df.groupby([COL_PDL, 'year_week', 'label'])[COL_PWR].apply(list).reset_index()
sequences['len'] = sequences[COL_PWR].apply(len)

# On filtre uniquement les semaines complètes
valid_seqs = sequences[sequences['len'] == SEQ_LEN].copy()
print(f"Nombre de séquences hebdomadaires valides : {len(valid_seqs)}")

# Séparation RP et RS
seqs_rp = valid_seqs[valid_seqs['label'] == 0][COL_PWR].tolist()
seqs_rs = valid_seqs[valid_seqs['label'] == 1][COL_PWR].tolist()

X_rp = np.array(seqs_rp)
X_rs = np.array(seqs_rs)

print(f"Shape X_rp (Résidence Principale) : {X_rp.shape}")
print(f"Shape X_rs (Résidence Secondaire) : {X_rs.shape}")

# %%
# =============================================================================
# 3. MISE À L'ÉCHELLE ET DATALOADERS
# =============================================================================
# Les GANs fonctionnent mieux avec des données entre -1 et 1
scaler_rp = MinMaxScaler(feature_range=(-1, 1))
X_rp_scaled = scaler_rp.fit_transform(X_rp)

scaler_rs = MinMaxScaler(feature_range=(-1, 1))
X_rs_scaled = scaler_rs.fit_transform(X_rs)

BATCH_SIZE = 64

tensor_rp = torch.tensor(X_rp_scaled, dtype=torch.float32)
dataset_rp = TensorDataset(tensor_rp)
dataloader_rp = DataLoader(dataset_rp, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

tensor_rs = torch.tensor(X_rs_scaled, dtype=torch.float32)
dataset_rs = TensorDataset(tensor_rs)
dataloader_rs = DataLoader(dataset_rs, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
# =============================================================================
# 4. ARCHITECTURE DU GAN (MLP Simple)
# =============================================================================
LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, seq_len),
            nn.Tanh() # Sortie entre -1 et 1
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # Sortie entre 0 (Fake) et 1 (Real)
        )

    def forward(self, seq):
        return self.model(seq)

# %%
# =============================================================================
# 5. FONCTION D'ENTRAÎNEMENT DU GAN
# =============================================================================
def train_gan(dataloader, num_epochs=200, lr=0.0002):
    generator = Generator(LATENT_DIM, SEQ_LEN).to(device)
    discriminator = Discriminator(SEQ_LEN).to(device)

    criterion = nn.BCELoss()
    # Optimizers (Adam souvent recommandé pour les GANs)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {'D_loss': [], 'G_loss': []}

    print("Début de l'entraînement...")
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        for i, (real_seqs,) in enumerate(dataloader):
            batch_size = real_seqs.size(0)
            real_seqs = real_seqs.to(device)

            # Labels pour la fonction de coût
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Perte sur les vraies séquences
            outputs_real = discriminator(real_seqs)
            d_loss_real = criterion(outputs_real, real_labels)
            
            # Perte sur les fausses séquences
            z = torch.randn((batch_size, LATENT_DIM)).to(device)
            fake_seqs = generator(z)
            outputs_fake = discriminator(fake_seqs.detach()) # Detach pour ne pas backprop dans G
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            # Mise à jour D
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Le générateur veut que le discriminateur croie que ses séquences sont vraies (label 1)
            outputs = discriminator(fake_seqs)
            g_loss = criterion(outputs, real_labels)
            
            # Mise à jour G
            g_loss.backward()
            optimizer_G.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Moyenne sur l'époque
        history['D_loss'].append(epoch_d_loss / len(dataloader))
        history['G_loss'].append(epoch_g_loss / len(dataloader))

        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | D_loss: {history['D_loss'][-1]:.4f} | G_loss: {history['G_loss'][-1]:.4f}")
            
    return generator, discriminator, history

# %%
# =============================================================================
# 6. ENTRAÎNEMENT DU GAN RP (Résidence Principale)
# =============================================================================
# On lance avec peu d'epochs par défaut pour tester. Mettre 500 ou 1000 pour de vrais résultats.
EPOCHS_TEST = 100 
print("--- Entraînement GAN RP ---")
gen_rp, disc_rp, hist_rp = train_gan(dataloader_rp, num_epochs=EPOCHS_TEST)

# %%
# =============================================================================
# 7. ENTRAÎNEMENT DU GAN RS (Résidence Secondaire)
# =============================================================================
print("\n--- Entraînement GAN RS ---")
gen_rs, disc_rs, hist_rs = train_gan(dataloader_rs, num_epochs=EPOCHS_TEST)

# %%
# =============================================================================
# 8. GÉNÉRATION ET VISUALISATION
# =============================================================================
def plot_generated_vs_real(generator, real_data, scaler, title=""):
    generator.eval()
    with torch.no_grad():
        # Générer 5 échantillons
        z = torch.randn((5, LATENT_DIM)).to(device)
        fake_scaled = generator(z).cpu().numpy()
        
    # Remise à l'échelle d'origine
    fake_curves = scaler.inverse_transform(fake_scaled)
    # Remplacer les valeurs < 0 par 0 (physiquement impossible d'avoir une conso négative)
    fake_curves = np.maximum(fake_curves, 0)
    
    # Prendre 5 vrais échantillons au hasard
    idx = np.random.randint(0, len(real_data), size=5)
    real_curves = real_data[idx]
    
    x_axis = np.arange(SEQ_LEN) / 2  # Axe des X en Heures (0 à 168)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i in range(5):
        axes[0].plot(x_axis, real_curves[i], alpha=0.7)
    axes[0].set_title(f"Réel ({title})")
    axes[0].set_xlabel("Heures dans la semaine")
    axes[0].set_ylabel("Puissance (kW)")
    
    for i in range(5):
        axes[1].plot(x_axis, fake_curves[i], alpha=0.7, linestyle='--')
    axes[1].set_title(f"Généré par GAN ({title})")
    axes[1].set_xlabel("Heures dans la semaine")
    
    plt.tight_layout()
    plt.show()

# Visualisation RP
plot_generated_vs_real(gen_rp, X_rp, scaler_rp, title="RP")

# Visualisation RS
plot_generated_vs_real(gen_rs, X_rs, scaler_rs, title="RS")

# Tracer les courbes de Loss pour vérifier que le GAN a convergé
plt.figure(figsize=(10, 4))
plt.plot(hist_rp['D_loss'], label='D_loss (RP)')
plt.plot(hist_rp['G_loss'], label='G_loss (RP)')
plt.title("Historique des Pertes (Loss) - GAN RP")
plt.xlabel("Epoch")
plt.legend()
plt.show()
