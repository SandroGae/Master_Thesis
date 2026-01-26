#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV\csv_triple_loss") 

def evaluate_all_runs(base_path):
    all_results = []

    # Suche alle CSV Dateien rekursiv
    csv_files = list(base_path.glob("**/log_*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {base_path} gefunden! Prüfe den Pfad.")
        return

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            if 'val_loss' not in df.columns:
                continue
                
            best_idx = df['val_loss'].idxmin()
            best_row = df.loc[best_idx]
            
            run_name = csv_path.stem.replace("log_", "")
            all_results.append({
                'run_name': run_name,
                'best_epoch': int(best_row['epoch']),
                'val_loss': best_row['val_loss'],
                'mae': best_row['val_mae_center'],
                'mse': best_row['val_mse_center'],
                'ssim': best_row['val_ssim_center']
            })
        except Exception as e:
            print(f"Fehler beim Lesen von {csv_path.name}: {e}")

    results_df = pd.DataFrame(all_results)

    # 2. NORMALISIERUNG (NCS Berechnung) - Bleibt mathematisch identisch
    def normalize(series, reverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min: return series * 0 + 1.0
        if reverse: 
            return (s_max - series) / (s_max - s_min)
        else: 
            return (series - s_min) / (s_max - s_min)

    results_df['n_mae']  = normalize(results_df['mae'], reverse=True)
    results_df['n_mse']  = normalize(results_df['mse'], reverse=True)
    results_df['n_ssim'] = normalize(results_df['ssim'], reverse=False)

    # 3. SUCCESS SCORE BERECHNEN
    results_df['success_score'] = (results_df['n_mae'] + results_df['n_mse'] + results_df['n_ssim']) / 3

    # Ranking erstellen
    results_df = results_df.sort_values('success_score', ascending=False).reset_index(drop=True)

    # 4. OUTPUT IN TXT DATEI
    output_file = base_path / "evaluation_summary.txt"
    with open(output_file, "w") as f:
        f.write("="*125 + "\n")
        f.write(f"MASTER THESIS EVALUATION - FOLDER: {base_path.name}\n")
        f.write(f"Ranked by NCS | Scaling in Table: MSE*100, MAE*10\n")
        f.write("="*125 + "\n\n")
        
        # Header angepasst: PSNR raus, MSE*100 rein, MAE*10 Label
        header = f"{'Rank':<5} {'Run Name':<60} {'Score':<8} {'MSE*100':<10} {'SSIM':<8} {'MAE*10':<10} {'Epoch':<5}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for i, row in results_df.iterrows():
            # Hier findet die Skalierung NUR für die Text-Ausgabe statt
            display_mse = row['mse'] * 100
            display_mae = row['mae'] * 10
            
            line = (f"{i+1:<5} {row['run_name'][:58]:<60} {row['success_score']:8.4f} "
                    f"{display_mse:10.6f} {row['ssim']:8.4f} {display_mae:10.6f} {int(row['best_epoch']):<5}\n")
            f.write(line)

    print(f"Evaluation abgeschlossen! {len(results_df)} Runs analysiert.")
    print(f"Zusammenfassung (MSE*100 / MAE*10) gespeichert in: {output_file}")

if __name__ == "__main__":
    evaluate_all_runs(BASE_DIR)