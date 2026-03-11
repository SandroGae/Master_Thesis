import numpy as np

# Lade eine beliebige NPZ-Datei aus deinem Ordner
data = np.load(r"C:\Users\sandr\VS_Master_Thesis\Confidence\npz_files\CARE_10_SEEDS\Eval_Confidence_MSE_seed42_S05.npz")

# Zeige alle Namen an, die in dieser Datei gespeichert sind
print("Vorhandene Keys in der NPZ-Datei:", data.files)