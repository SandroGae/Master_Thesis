import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 1. Setup
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv" 

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# 2. Area Ratios & Penalized-Logik
for d in ["H", "K", "L"]:
    df[f"Area_{d}"] = df[f"Amp_pr_{d}"] / df[f"Amp_gt_{d}"]
    df[f"Area_{d}"] = df[f"Area_{d}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

df["Area_Row_Mean"] = df[["Area_H", "Area_K", "Area_L"]].mean(axis=1)

# --------------------------------------------------
# 3. Detaillierte Serien-Analyse pro Top-Modell
# --------------------------------------------------

# Hilfsfunktion für die Performance-Metriken pro Serie
def get_series_stats(group):
    return pd.Series({
        'Mean_All_Series': group['Area_Row_Mean'].mean(),
        'Best_Series_ID': group.loc[group['Area_Row_Mean'].idxmax(), 'SeriesID'],
        'Best_Series_Val': group['Area_Row_Mean'].max(),
        'Worst_Series_ID': group.loc[group['Area_Row_Mean'].idxmin(), 'SeriesID'],
        'Worst_Series_Val': group['Area_Row_Mean'].min(),
        'Std_Dev_Series': group['Area_Row_Mean'].std() # Maß für die Konsistenz
    })

# A: Analyse für Hyperparameter-Punkte (über alle Seeds & Serien)
point_stats = df.groupby('Point').apply(get_series_stats).reset_index()
point_ranking = point_stats.sort_values(by="Mean_All_Series", ascending=False).reset_index(drop=True)
point_ranking['Rank'] = range(len(point_ranking), 0, -1)

# B: Analyse für Einzel-Modelle (Point + Seed über alle Serien)
seed_stats = df.groupby(['Point', 'Seed']).apply(get_series_stats).reset_index()
seed_ranking = seed_stats.sort_values(by="Mean_All_Series", ascending=False).reset_index(drop=True)
seed_ranking['Rank'] = range(len(seed_ranking), 0, -1)

# --------------------------------------------------
# 4. Output & Export
# --------------------------------------------------

print("\n" + "="*110)
print(f"{'TOP 10 PUNKTE: PERFORMANCE-VERGLEICH ÜBER 30 SERIEN':^110}")
print("="*110)
header = f"{'Point':<6} | {'Schnitt':<10} | {'Beste Serie (S-ID/Val)':<25} | {'Schlechteste (S-ID/Val)':<25} | {'Stabilität (Std)':<10}"
print(header)
print("-" * len(header))

for _, row in point_ranking.head(43).iterrows():
    best_str = f"S-{int(row['Best_Series_ID']):<2} ({row['Best_Series_Val']:.3f})"
    worst_str = f"S-{int(row['Worst_Series_ID']):<2} ({row['Worst_Series_Val']:.3f})"
    print(f"{int(row['Point']):<6} | {row['Mean_All_Series']:<10.4f} | {best_str:<25} | {worst_str:<25} | {row['Std_Dev_Series']:<10.4f}")

print("\n" + "="*110)
print(f"{'TOP 10 EINZEL-MODELLE: PERFORMANCE-VERGLEICH ÜBER 30 SERIEN':^110}")
print("="*110)
header_s = f"{'Pt/Seed':<10} | {'Schnitt':<10} | {'Beste Serie (S-ID/Val)':<25} | {'Schlechteste (S-ID/Val)':<25} | {'Stabilität (Std)':<10}"
print(header_s)
print("-" * len(header_s))

for _, row in seed_ranking.head(43).iterrows():
    pt_sd = f"{int(row['Point'])}/{int(row['Seed'])}"
    best_str = f"S-{int(row['Best_Series_ID']):<2} ({row['Best_Series_Val']:.3f})"
    worst_str = f"S-{int(row['Worst_Series_ID']):<2} ({row['Worst_Series_Val']:.3f})"
    print(f"{pt_sd:<10} | {row['Mean_All_Series']:<10.4f} | {best_str:<25} | {worst_str:<25} | {row['Std_Dev_Series']:<10.4f}")

# Export der vollen Matrix für die Thesis
point_ranking.to_csv(SCRIPT_DIR / "Detailed_Point_Series_Performance.csv", index=False)
seed_ranking.to_csv(SCRIPT_DIR / "Detailed_Seed_Series_Performance.csv", index=False)

print(f"\nErfolg: Detaillierte Serien-Berichte in {SCRIPT_DIR} gespeichert.")