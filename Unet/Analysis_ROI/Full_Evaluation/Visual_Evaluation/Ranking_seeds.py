import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 1. Setup
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv"

if not CSV_FILE.exists():
    CSV_FILE = SCRIPT_DIR.parent / "Full_Evaluation_Results_Extended.csv"
    if not CSV_FILE.exists():
        raise FileNotFoundError("Datei nicht gefunden! Bitte prüfe den Pfad.")

df = pd.read_csv(CSV_FILE)

# Strenge Qualitäts-Grenzwerte
R2_RATIO_MIN = 0.6269
RMSE_RATIO_MAX = 1.0949
SERIES_TO_CHECK = 22

# 2. Fail-Spalten pro Richtung berechnen
for d in ["H", "K", "L"]:
    df[f"Area_{d}"] = (df[f"Amp_pr_{d}"] * df[f"Sig_pr_{d}"]) / (df[f"Amp_gt_{d}"] * df[f"Sig_gt_{d}"])

    r2_ratio = df[f"R2_pr_{d}"] / df[f"R2_gt_{d}"]
    rmse_ratio = df[f"RMSE_pr_{d}"] / df[f"RMSE_gt_{d}"]

    df[f"Reason_NaN_{d}"] = df[f"Area_{d}"].isna()
    df[f"Reason_R2_{d}"] = r2_ratio < R2_RATIO_MIN
    df[f"Reason_RMSE_{d}"] = rmse_ratio > RMSE_RATIO_MAX

    df[f"Fail_{d}"] = df[f"Reason_NaN_{d}"] | df[f"Reason_R2_{d}"] | df[f"Reason_RMSE_{d}"]

# 3. Nur Serie 22
df22 = df[df["SeriesID"] == SERIES_TO_CHECK].copy()

# 4. Verworfene Richtungen als einzelne Zeilen sammeln
rows = []

for _, row in df22.iterrows():
    for d in ["H", "K", "L"]:
        if row[f"Fail_{d}"]:
            reasons = []
            if row[f"Reason_NaN_{d}"]:
                reasons.append("NaN")
            if row[f"Reason_R2_{d}"]:
                reasons.append("R2")
            if row[f"Reason_RMSE_{d}"]:
                reasons.append("RMSE")

            rows.append({
                "Point": int(row["Point"]),
                "Seed": int(row["Seed"]),
                "SeriesID": int(row["SeriesID"]),
                "Direction": d,
                "Area": row[f"Area_{d}"],
                "R2_pr": row[f"R2_pr_{d}"],
                "R2_gt": row[f"R2_gt_{d}"],
                "RMSE_pr": row[f"RMSE_pr_{d}"],
                "RMSE_gt": row[f"RMSE_gt_{d}"],
                "Reason": ",".join(reasons)
            })

rejected_df = pd.DataFrame(rows).sort_values(["Point", "Seed", "Direction"])

# 5. Ausgabe
print("\n" + "=" * 120)
print(f"VERWORFENE PLOTS IN SERIE {SERIES_TO_CHECK}")
print("=" * 120)

if rejected_df.empty:
    print("Keine verworfenen Plots in dieser Serie gefunden.")
else:
    print(rejected_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("ANZAHL VERWORFENER RICHTUNGEN PRO POINT")
    print("=" * 120)
    print(rejected_df.groupby("Point").size().sort_values(ascending=False).to_string())

    print("\n" + "=" * 120)
    print("ANZAHL VERWORFENER RICHTUNGEN PRO POINT UND SEED")
    print("=" * 120)
    print(rejected_df.groupby(["Point", "Seed"]).size().to_string())