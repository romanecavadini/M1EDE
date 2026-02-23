import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

raw_dataset = Path.cwd() / ".." /"data" / "RES2-6-9.csv"
df = pd.read_csv(raw_dataset, 
                 sep=';',      # Le séparateur de colonnes
                 decimal='.') # Optionnel : transforme la 2ème colonne en vraie date
print(df.head(), df.info())