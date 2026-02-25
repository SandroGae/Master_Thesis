from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Setup & Pfade
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR.parent / "Evaluation_Metrics_H_K_L" / "Full_Evaluation_Results.csv"

# Dateinamen wie gewünscht
TXT_BEST = SCRIPT_DIR / "Full_Evaluation_BEST.txt"
TXT_AVG  = SCRIPT_DIR / "Full_Evaluation_AVERAGE.txt"

df = pd.read_csv(CSV_FILE)

# --- SPALTEN-FIX ---
metrics_cols = [
    "Area_H", "SBR_H", "Shift_H", 
    "Area_K", "SBR_K", "Shift_K", 
    "Area_L", "SBR_L", "Shift_L"
]

for col in metrics_cols + ["Alpha", "Beta", "Seed", "Point"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- 3D-DURCHSCHNITT PRO SERIE ---
df['Area_3D'] = df[['Area_H', 'Area_K', 'Area_L']].mean(axis=1)
df['Shift_3D'] = df[['Shift_H', 'Shift_K', 'Shift_L']].mean(axis=1)

# ======================================================================
# TEIL 1: Full_Evaluation_BEST.txt (Unverändert gelassen)
# ======================================================================
run_ranking = df.groupby(['Point', 'Seed', 'Alpha', 'Beta']).agg(
    Run_Area_Mean=('Area_3D', 'mean'),
    Run_Shift_Mean=('Shift_3D', 'mean'),
    IsClean=('IsClean', 'all')
).reset_index()

all_runs_sorted = run_ranking.sort_values('Run_Area_Mean', ascending=False).reset_index(drop=True)

with open(TXT_BEST, "w", encoding="utf-8") as f:
    f.write("="*95 + "\n")
    f.write("COMPLETE RUN RANKING (Sorted by 3D-Averaged AreaRatio)\n")
    f.write(f"Total Runs evaluated: {len(all_runs_sorted)}\n")
    f.write("One Run = Average of 10 Series (S5-S50) and 3 Directions (H,K,L)\n")
    f.write("="*95 + "\n\n")

    header = f"{'Rank':<6} {'Point':<7} {'Seed':<6} {'Alpha':<8} {'Beta':<8} {'MeanArea':<12} {'AvgShift':<10} {'Clean'}\n"
    f.write(header)
    f.write("-" * 95 + "\n")
    
    for i, row in all_runs_sorted.iterrows():
        clean_str = "YES" if row.IsClean else "NO"
        f.write(f"{i+1:<6} P{int(row.Point):02d} {int(row.Seed):<6} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                f"{row.Run_Area_Mean:<12.4f} {row.Run_Shift_Mean:<10.2f} {clean_str}\n")

    f.write("\n" + "="*95 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*95 + "\n")

# ======================================================================
# TEIL 2: Full_Evaluation_AVERAGE.txt (Mittelwert pro Punkt über 10 Seeds)
# ======================================================================
# Kopie für Average-Berechnung (wegen Alpha=1.0 Fix)
df_avg = df.copy()
mask_a1 = (df_avg["Alpha"] == 1.0)
df_avg.loc[mask_a1, "Beta"] = 0.0

point_stats = df_avg.groupby(['Point', 'Alpha', 'Beta']).agg(
    Mean_Area_3D=('Area_3D', 'mean'),
    Std_Area_3D=('Area_3D', 'std'),
    Mean_Shift_3D=('Shift_3D', 'mean'),
    Total_Runs=('IsClean', 'count'),
    Success_Runs=('IsClean', lambda x: x.sum())
).reset_index()

point_stats['SuccessRate'] = (point_stats['Success_Runs'] / point_stats['Total_Runs']) * 100
point_ranking = point_stats.sort_values('Mean_Area_3D', ascending=False).reset_index(drop=True)

with open(TXT_AVG, "w", encoding="utf-8") as f:
    f.write("="*105 + "\n")
    f.write("HYPERPARAMETER POINT AVERAGES (Ranked by MAX AreaRatio)\n")
    f.write("Each Point represents the average of 10 Seeds and 3 Directions (H,K,L)\n")
    f.write("="*105 + "\n\n")

    header = f"{'Rank':<6} {'Point':<7} {'Alpha':<8} {'Beta':<8} {'MeanArea':<12} {'StdDev':<10} {'AvgShift':<10} {'SuccessRate'}\n"
    f.write(header)
    f.write("-" * 105 + "\n")
    
    for i, row in point_ranking.iterrows():
        f.write(f"{i+1:<6} P{int(row.Point):02d} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                f"{row.Mean_Area_3D:<12.4f} {row.Std_Area_3D:<10.4f} {row.Mean_Shift_3D:<10.2f} {row.SuccessRate:>10.1f}%\n")

    f.write("\n" + "="*105 + "\n")
    f.write("INTERPRETATION:\n")
    f.write("- MeanArea: Higher values indicate stronger signal reconstruction relative to GT.\n")
    f.write("- StdDev: Reproducibility across seeds (lower is more stable).\n")
    f.write("- Alpha=1.0: Results consolidated at Beta=0.0.\n")
    f.write("="*105 + "\n")

print(f"Erfolg! Zwei Dateien wurden erstellt:")
print(f"1. Best Runs: {TXT_BEST.name}")
print(f"2. Point Averages: {TXT_AVG.name}")