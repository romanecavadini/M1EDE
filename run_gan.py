import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cpu")  # CPU pour éviter le crash MPS

DATA_PATH   = Path("data/export.csv")
LABELS_PATH = Path("data/RES2-6-9-labels.csv")
OUTPUT_PATH = Path("data/gan_curves.pkl")

# Hyperparamètres
STEPS_PER_HOUR = 2
HOURS_PER_DAY  = 24
DAYS_PER_WEEK  = 7
SEQ_LEN        = STEPS_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK  # 336
LATENT_DIM     = 100
EPOCHS         = 100
BATCH_SIZE     = 64

# ── Chargement ──
print("⏳ Chargement des données...")
raw = pd.read_csv(DATA_PATH, sep=',')
raw = raw.rename(columns={"id": "pdl_id", "horodate": "datetime", "valeur": "p_kw"})
raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
if raw["datetime"].dt.tz is None:
    raw["datetime"] = raw["datetime"].dt.tz_localize("Europe/Paris")
else:
    raw["datetime"] = raw["datetime"].dt.tz_convert("Europe/Paris")
df = raw.dropna(subset=["pdl_id", "datetime", "p_kw"]).copy()
df["p_kw"] = pd.to_numeric(df["p_kw"], errors="coerce")
df = df.dropna(subset=["p_kw"])

df_labels = pd.read_csv(LABELS_PATH, sep=';')
df_labels = df_labels.rename(columns={"id": "pdl_id"})
df = df.merge(df_labels[["pdl_id", "label"]], on="pdl_id", how="inner")

print(f"✅ {len(df)} lignes, {df['pdl_id'].nunique()} clients")

# ── Extraction séquences hebdomadaires ──
print("⏳ Extraction des séquences...")
df["year_week"] = df["datetime"].dt.strftime("%G-%V")
df = df.sort_values(["pdl_id", "datetime"])
sequences = df.groupby(["pdl_id", "year_week", "label"])["p_kw"].apply(list).reset_index()
sequences["len"] = sequences["p_kw"].apply(len)
valid_seqs = sequences[sequences["len"] == SEQ_LEN].copy()
print(f"✅ {len(valid_seqs)} séquences valides")

seqs_rp = valid_seqs[valid_seqs["label"] == 0]["p_kw"].tolist()
seqs_rs = valid_seqs[valid_seqs["label"] == 1]["p_kw"].tolist()
X_rp = np.array(seqs_rp)
X_rs = np.array(seqs_rs)

# ── Scalers ──
scaler_rp = MinMaxScaler(feature_range=(-1, 1))
scaler_rs = MinMaxScaler(feature_range=(-1, 1))
X_rp_scaled = scaler_rp.fit_transform(X_rp)
X_rs_scaled = scaler_rs.fit_transform(X_rs)

# ── Architecture GAN ──
class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 512),        nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
            nn.Linear(512, 1024),       nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
            nn.Linear(1024, seq_len),   nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_len, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2),
            nn.Linear(256, 1),       nn.Sigmoid()
        )
    def forward(self, seq):
        return self.model(seq)

def train_gan(X_scaled, num_epochs=EPOCHS):
    tensor   = torch.tensor(X_scaled, dtype=torch.float32)
    dataset  = TensorDataset(tensor)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator     = Generator(LATENT_DIM, SEQ_LEN).to(device)
    discriminator = Discriminator(SEQ_LEN).to(device)
    criterion     = nn.BCELoss()
    opt_G = optim.Adam(generator.parameters(),     lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for (real_seqs,) in loader:
            bs = real_seqs.size(0)
            real_seqs = real_seqs.to(device)
            real_labels = torch.ones(bs, 1).to(device)
            fake_labels = torch.zeros(bs, 1).to(device)

            # Train D
            opt_D.zero_grad()
            z = torch.randn(bs, LATENT_DIM).to(device)
            fake_seqs = generator(z)
            d_loss = (criterion(discriminator(real_seqs), real_labels) +
                      criterion(discriminator(fake_seqs.detach()), fake_labels)) / 2
            d_loss.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            g_loss = criterion(discriminator(fake_seqs), real_labels)
            g_loss.backward()
            opt_G.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} — D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    return generator

# ── Entraînement ──
print("\n--- Entraînement GAN Résidence Principale ---")
gen_rp = train_gan(X_rp_scaled)

print("\n--- Entraînement GAN Résidence Secondaire ---")
gen_rs = train_gan(X_rs_scaled)

# ── Génération des courbes ──
def generate_curves(generator, scaler, n=10):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM).to(device)
        fake_scaled = generator(z).cpu().numpy()
    curves = scaler.inverse_transform(fake_scaled)
    return np.maximum(curves, 0)

print("\n⏳ Génération des courbes...")
curves_rp_generated = generate_curves(gen_rp, scaler_rp, n=10)
curves_rs_generated = generate_curves(gen_rs, scaler_rs, n=10)

# ── Sauvegarde ──
output = {
    "rp": {
        "real":      X_rp[:10],
        "generated": curves_rp_generated,
        "seq_len":   SEQ_LEN
    },
    "rs": {
        "real":      X_rs[:10],
        "generated": curves_rs_generated,
        "seq_len":   SEQ_LEN
    }
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(output, f)

print(f"\n✅ Courbes sauvegardées dans {OUTPUT_PATH}")
