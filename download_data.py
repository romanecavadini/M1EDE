import gdown
from pathlib import Path
import shutil

Path("data").mkdir(exist_ok=True)

FILE_ID = "1aZUOVjMTAhSegI70kPmFjUtPWl644FhT"

if not Path("data/export.csv").exists():
    print("⏳ Téléchargement export.csv depuis Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "data/export.csv", quiet=False)
    print("✅ export.csv téléchargé !")
else:
    print("✅ export.csv déjà présent.")

if not Path("data/RES2-6-9.csv").exists():
    print("⏳ Copie vers RES2-6-9.csv...")
    shutil.copy("data/export.csv", "data/RES2-6-9.csv")
    print("✅ RES2-6-9.csv créé !")
else:
    print("✅ RES2-6-9.csv déjà présent.")
