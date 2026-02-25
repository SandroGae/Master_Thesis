import pandas as pd
import numpy as np
from pathlib import Path
import os

# =====================================================
# 1. SETUP (NUTZT ORDNER DES SKRIPTS)
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR / "Image_Quality_Metrics_Testset.csv"
OUTPUT_TXT = SCRIPT_DIR / "Model_Ranking_Clipped_Final.txt"

print(f"> Lade CSV: {FILE_PATH}")

if not FILE_PATH.exists():
    print(f"!!! FEHLER: Datei {FILE_PATH.name} nicht gefunden!")
    print(f"Stelle sicher, dass die CSV im Ordner liegt: {SCRIPT_DIR}")
    exit()

df = pd.read_csv(FILE_PATH)

# =====================================================
# 2. SCORING LOGIK (BASIS: CLIPPED METRIKEN)
# =====================================================
def calculate_relative_score(series, direction='higher_is_better'):
    """
    Lineare relative Skalierung: Bestwert = 1.0.
    A score of 0.5 means twice as bad as the best.
    """
    if direction == 'higher_is_better':
        # SSIM: Höher ist besser
        best_val = series.max()
        return series / (best_val + 1e-12)
    else:
        # MAE/MSE: Niedriger ist besser
        best_val = series.min()
        return best_val / (series + 1e-12)

# Berechnung der relativen Scores für MAE, MSE und SSIM (Clipped)
df['score_MAE'] = calculate_relative_score(df['MAE_clipped'], direction='lower_is_better')
df['score_MSE'] = calculate_relative_score(df['MSE_clipped'], direction='lower_is_better')
df['score_SSIM'] = calculate_relative_score(df['SSIM_clipped'], direction='higher_is_better')

# Durchschnittsscore (jede Metrik zählt genau 1/3)
df['QualityScore'] = (df['score_MAE'] + df['score_MSE'] + df['score_SSIM']) / 3.0

# Ranking nach dem QualityScore sortieren
df_ranked = df.sort_values('QualityScore', ascending=False).reset_index(drop=True)

# =====================================================
# 3. SPEICHERN ALS TEXTDATEI
# =====================================================
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("="*125 + "\n")
    f.write("FINAL MODEL RANKING (Based on CLIPPED Keras-Match Metrics)\n")
    f.write("Logic: Error-Score = Min_Value/Value | SSIM-Score = Value/Max_Value\n")
    f.write("A score of 1.0 means this model is the best in its category.\n")
    f.write("-" * 125 + "\n")
    
    header = f"{'Rank':<6} {'Point':<6} {'Seed':<6} {'Alpha':<8} {'Beta':<8} {'Score':<10} {'MAE_cl':<10} {'MSE_cl':<10} {'SSIM_cl':<10} {'PSNR_cl':<8}\n"
    f.write(header)
    f.write("-" * 125 + "\n")
    
    for i, row in df_ranked.iterrows():
        f.write(f"{i+1:<6} P{int(row.Point):02d} {int(row.Seed):<6} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                f"{row.QualityScore:<10.4f} {row.MAE_clipped:<10.4f} {row.MSE_clipped:<10.6f} "
                f"{row.SSIM_clipped:<10.4f} {row.PSNR_clipped:<8.2f}\n")

print(f"\n>>> FERTIG! Ranking erstellt: {OUTPUT_TXT.name}")
print(f"Top-Modell (Clipped): Point {int(df_ranked.iloc[0]['Point'])} (Seed {int(df_ranked.iloc[0]['Seed'])})")
print(f"Bestes PSNR (Clipped): {df_ranked.iloc[0]['PSNR_clipped']:.2f} dB")