import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# SETUP
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = SCRIPT_DIR / "Image_Quality_Metrics_Testset.csv"

def process_and_save(df_raw, mode):
    m = mode.lower()
    mae_col, mse_col, ssim_col, psnr_col = f'MAE_{m}', f'MSE_{m}', f'SSIM_{m}', f'PSNR_{m}'
    
    display_name = "Clipped" if m == "clipped" else "Unclipped"
    txt_file = SCRIPT_DIR / f"Model_Ranking_{display_name}_Final.txt"
    csv_file = SCRIPT_DIR / f"Model_Ranking_{display_name}_Final-csv.csv"

    # --- SCHRITT 1: Durchschnittswerte der 43 Punkte berechnen ---
    avg_df = df_raw.groupby(['Point', 'Alpha', 'Beta']).agg({
        mae_col: 'mean', mse_col: 'mean', ssim_col: 'mean', psnr_col: 'mean'
    }).reset_index()

    # --- SCHRITT 2: Champions NUR unter den 43 Durchschnitten finden ---
    ref_mae = avg_df[mae_col].min()
    ref_mse = avg_df[mse_col].min()
    ref_ssim = avg_df[ssim_col].max()

    # --- SCHRITT 3: Lineare Verhältnisskalierung (Best = 1.0) ---
    avg_df['score_MAE'] = ref_mae / avg_df[mae_col]
    avg_df['score_MSE'] = ref_mse / avg_df[mse_col]
    avg_df['score_SSIM'] = avg_df[ssim_col] / ref_ssim
    
    # --- SCHRITT 4: QualityScore (Summe / 3) ---
    avg_df['QualityScore'] = (avg_df['score_MAE'] + avg_df['score_MSE'] + avg_df['score_SSIM']) / 3.0
    averaged_ranked = avg_df.sort_values('QualityScore', ascending=False).reset_index(drop=True)

    # --- ANALOG FÜR SEKTION 2 (Individuelle Seeds gegen Individual-Champions) ---
    ind_df = df_raw.copy()
    i_ref_mae, i_ref_mse, i_ref_ssim = ind_df[mae_col].min(), ind_df[mse_col].min(), ind_df[ssim_col].max()
    ind_df['QualityScore'] = ( (i_ref_mae/ind_df[mae_col]) + (i_ref_mse/ind_df[mse_col]) + (ind_df[ssim_col]/i_ref_ssim) ) / 3.0
    individual_ranked = ind_df.sort_values('QualityScore', ascending=False).reset_index(drop=True)

    # --- SCHREIBEN ---
    with open(txt_file, "w", encoding="utf-8") as f:
        line = "="*130 + "\n"
        f.write(line)
        f.write(f"MODEL RANKING REPORT - MODE: {mode.upper()}\n")
        f.write("Logic: For each metric, the best POINT-AVERAGE is 100% (1.0). Others scale linearly.\n")
        f.write("Final Score = (MAE_Score + MSE_Score + SSIM_Score) / 3\n")
        f.write(line + "\n")

        f.write("SECTION 1: AVERAGED PERFORMANCE PER POINT (Mean of 10 Seeds)\n")
        f.write("-" * 130 + "\n")
        header = f"{'Rank':<6} {'Point':<12} {'Alpha':<8} {'Beta':<8} {'Score':<10} {'MAE_avg':<10} {'MSE_avg':<12} {'SSIM_avg':<10} {'PSNR_avg':<8}\n"
        f.write(header)
        f.write("-" * 130 + "\n")
        for i, row in averaged_ranked.iterrows():
            f.write(f"{i+1:<6} Point {int(row.Point):02d}    {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                    f"{row.QualityScore:<10.4f} {row[mae_col]:<10.4f} {row[mse_col]:<12.6f} "
                    f"{row[ssim_col]:<10.4f} {row[psnr_col]:<8.2f}\n")

        f.write("\n\n" + line)
        f.write("SECTION 2: INDIVIDUAL MODEL SEED RANKING (Against Individual Bests)\n")
        f.write("-" * 130 + "\n")
        header_ind = f"{'Rank':<6} {'Point':<6} {'Seed':<6} {'Alpha':<8} {'Beta':<8} {'Score':<10} {'MAE':<10} {'MSE':<12} {'SSIM':<10} {'PSNR':<8}\n"
        f.write(header_ind)
        f.write("-" * 130 + "\n")
        for i, row in individual_ranked.iterrows():
            f.write(f"{i+1:<6} P{int(row.Point):02d} {int(row.Seed):<6} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                    f"{row.QualityScore:<10.4f} {row[mae_col]:<10.4f} {row[mse_col]:<12.6f} "
                    f"{row[ssim_col]:<10.4f} {row[psnr_col]:<8.2f}\n")

    averaged_ranked.to_csv(csv_file, index=False)
    print(f">>> Erstellt: {txt_file.name} & {csv_file.name}")

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    for m in ['clipped', 'unclipped']:
        process_and_save(df, m)