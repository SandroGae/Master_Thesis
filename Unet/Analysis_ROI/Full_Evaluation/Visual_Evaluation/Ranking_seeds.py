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
        raise FileNotFoundError(f"Datei nicht gefunden! Bitte prüfe den Pfad.")

df = pd.read_csv(CSV_FILE)

# Strenge Qualitäts-Grenzwerte
R2_RATIO_MIN = 0.6269
RMSE_RATIO_MAX = 1.0949

# --- 2. Metriken berechnen ---
for d in ["H", "K", "L"]:
    # Physikalische Fläche
    df[f"Area_{d}"] = (df[f"Amp_pr_{d}"] * df[f"Sig_pr_{d}"]) / (df[f"Amp_gt_{d}"] * df[f"Sig_gt_{d}"])
    
    # Check gegen die strengen Grenzwerte
    r2_ratio = df[f"R2_pr_{d}"] / df[f"R2_gt_{d}"]
    rmse_ratio = df[f"RMSE_pr_{d}"] / df[f"RMSE_gt_{d}"]
    
    # Markiere Ausfälle
    df[f"Fail_{d}"] = (df[f"Area_{d}"].isna()) | (r2_ratio < R2_RATIO_MIN) | (rmse_ratio > RMSE_RATIO_MAX)
    
    # Bereinigte Fläche (Filter-Punkte werden als 0 gewertet)
    df[f"Area_Clean_{d}"] = df[f"Area_{d}"].where(~df[f"Fail_{d}"], 0.0)

# Zeilen-Statistik
df["Row_Failed"] = df[["Fail_H", "Fail_K", "Fail_L"]].any(axis=1)
df["Area_Row_Mean"] = df[["Area_Clean_H", "Area_Clean_K", "Area_Clean_L"]].mean(axis=1)

# --- 3. Robustheits-Aggregation pro Serie ---
series_stats = df.groupby('SeriesID').agg(
    Filter_Rate=('Row_Failed', 'mean'),
    Mean_Area=('Area_Row_Mean', 'mean'),
    Std_Area=('Area_Row_Mean', 'std')
).reset_index()

# 4. Penalty Score Berechnung
series_stats['Bias'] = abs(series_stats['Mean_Area'] - 1.0)

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

series_stats['Score_Filter'] = normalize(series_stats['Filter_Rate'])
series_stats['Score_Bias'] = normalize(series_stats['Bias'])
series_stats['Score_Std'] = normalize(series_stats['Std_Area'])

series_stats['Penalty_Score'] = (
    series_stats['Score_Filter'] * 0.5 + 
    series_stats['Score_Bias'] * 0.3 + 
    series_stats['Score_Std'] * 0.2
)

# Ranking erstellen
robust_ranking = series_stats.sort_values(by='Penalty_Score', ascending=False)

# --- 5. Output ---
print("\n" + "="*140)
print(f"{'VOLLSTÄNDIGES SERIEN-RANKING (VON SCHLECHT NACH GUT)':^140}")
print("="*140)
header = f"{'SeriesID':<10} | {'Filter-Rate':<12} | {'Schnitt Area':<15} | {'Std Dev':<12} | {'Bias (zu 1.0)':<15} | {'PENALTY SCORE':<15}"
print(header)
print("-" * len(header))

top_4_threshold = robust_ranking['Penalty_Score'].nlargest(4).min()

for _, row in robust_ranking.iterrows():
    prefix = ">>> " if row['Penalty_Score'] >= top_4_threshold else "    "
    print(f"{prefix}S-{int(row['SeriesID']):<6} | {row['Filter_Rate']*100:>10.1f}% | {row['Mean_Area']:>15.4f} | {row['Std_Area']:>12.4f} | {row['Bias']:>15.4f} | {row['Penalty_Score']:>15.4f}")

print("-" * len(header))
print(f"INFO: Serien mit '>>>' sind deine aktuellen Top-4 Kick-Kandidaten.")

# Kurze statistische Zusammenfassung der "Guten"
good_series = robust_ranking.tail(5)
print("\nZUM VERGLEICH - DEINE 5 BESTEN SERIEN (MUSTERSCHÜLER):")
print(good_series[['SeriesID', 'Filter_Rate', 'Mean_Area', 'Penalty_Score']].to_string(index=False))


# --------------------------------------------------
# 6. Detail-Analyse: Amplitude & Sigma Ratios
# --------------------------------------------------

for d in ["H", "K", "L"]:
    # Ratios berechnen
    df[f"Amp_Ratio_{d}"] = df[f"Amp_pr_{d}"] / df[f"Amp_gt_{d}"]
    df[f"Sig_Ratio_{d}"] = df[f"Sig_pr_{d}"] / df[f"Sig_gt_{d}"]
    
    # Nur valide Punkte nutzen (Filter-Ausfälle ignorieren, damit sie den Schnitt nicht verfälschen)
    df[f"Amp_Ratio_{d}"] = df[f"Amp_Ratio_{d}"].where(~df[f"Fail_{d}"], np.nan)
    df[f"Sig_Ratio_{d}"] = df[f"Sig_Ratio_{d}"].where(~df[f"Fail_{d}"], np.nan)

# Durchschnittliche Ratios pro Zeile
df["Amp_Ratio_Mean"] = df[["Amp_Ratio_H", "Amp_Ratio_K", "Amp_Ratio_L"]].mean(axis=1)
df["Sig_Ratio_Mean"] = df[["Sig_Ratio_Mean_H" if "Sig_Ratio_Mean_H" in df else "Sig_Ratio_H", 
                           "Sig_Ratio_K", "Sig_Ratio_L"]].mean(axis=1) # Korrektur falls Namen variieren

# Aggregation pro Serie
shape_stats = df.groupby('SeriesID').agg(
    Avg_Amp_Ratio=('Amp_Ratio_Mean', 'mean'),
    Avg_Sig_Ratio=('Sig_Ratio_Mean', 'mean')
).reset_index()

# Ranking erstellen: 
# "Schlecht" = Kleine Amplitude (Ratio gegen 0) UND Grosses Sigma (Ratio > 1.0)
shape_stats['Shape_Score'] = (1.0 - shape_stats['Avg_Amp_Ratio']) + (shape_stats['Avg_Sig_Ratio'] - 1.0)
shape_ranking = shape_stats.sort_values(by='Shape_Score', ascending=False)

print("\n" + "="*125)
print(f"{'SHAPE-ANALYSE: RATIO VON AMPLITUDE UND SIGMA (PR / GT)':^125}")
print("="*125)
header_shape = f"{'SeriesID':<10} | {'Amp-Ratio (Ziel: 1.0)':<25} | {'Sigma-Ratio (Ziel: 1.0)':<25} | {'Shape-Score (Höher=Schlechter)':<15}"
print(header_shape)
print("-" * len(header_shape))

for _, row in shape_ranking.iterrows():
    # Markierung für die extremen Fälle
    note = " <!!>" if row['Avg_Amp_Ratio'] < 0.6 or row['Avg_Sig_Ratio'] > 1.4 else ""
    print(f"S-{int(row['SeriesID']):<8} | {row['Avg_Amp_Ratio']:>20.4f}      | {row['Avg_Sig_Ratio']:>22.4f}       | {row['Shape_Score']:>15.4f} {note}")

print("-" * len(header_shape))
print("HINWEIS: Amp-Ratio < 1 bedeutet Unterschätzung. Sigma-Ratio > 1 bedeutet 'verschmiertes' Signal.")