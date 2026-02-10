#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
# Pfad zu deinem neuen Sammelordner für die RERUN-CSVs
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV\csv_RERUN")

# Name der neuen Output-Datei
OUTPUT_FILENAME = "combined_evaluation_summary_RERUN_ONLY.txt"

def evaluate_all_runs(base_path):
    all_results = []

    # rglob("*.csv") findet alle CSVs in allen Unterordnern von csv_RERUN
    csv_files = list(base_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {base_path} gefunden! Prüfe, ob der Pfad stimmt.")
        return

    print(f"Gefundene Dateien: {len(csv_files)}")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # Wir definieren die Spaltennamen flexibel, falls sie mal variieren
            # (Standardmäßig bei deinen DeepScan-Runs: val_loss, val_mae_center, etc.)
            col_loss = 'val_loss'
            col_mae  = 'val_mae_center' if 'val_mae_center' in df.columns else 'val_mae'
            col_mse  = 'val_mse_center' if 'val_mse_center' in df.columns else 'val_mse'
            col_ssim = 'val_ssim_center' if 'val_ssim_center' in df.columns else 'val_ssim'

            if col_loss not in df.columns:
                # Falls die CSV keine Validierungsdaten hat, überspringen
                continue
                
            # Finde die Epoche mit dem niedrigsten Validation Loss
            best_idx = df[col_loss].idxmin()
            best_row = df.loc[best_idx]
            
            # run_name ist einfach der Dateiname ohne .csv
            run_name = csv_path.stem
            
            all_results.append({
                'run_name': run_name,
                'origin': csv_path.parent.name, # Der Ordner-Zeitstempel vom Server
                'best_epoch': int(best_row['epoch']) if 'epoch' in best_row else best_idx,
                'val_loss': best_row[col_loss],
                'mae': best_row[col_mae],
                'mse': best_row[col_mse],
                'ssim': best_row[col_ssim]
            })
        except Exception as e:
            print(f"Fehler beim Lesen von {csv_path.name}: {e}")

    if not all_results:
        print("Keine gültigen Daten extrahiert. Prüfe die Spaltennamen in den CSVs.")
        return

    results_df = pd.DataFrame(all_results)

    # --- NORMALISIERUNG ---
    def normalize(series, reverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min: return series * 0 + 1.0
        if reverse: 
            return (s_max - series) / (s_max - s_min)
        else: 
            return (series - s_min) / (s_max - s_min)

    # Je höher der Score, desto besser das Modell
    results_df['n_mae']  = normalize(results_df['mae'], reverse=True)
    results_df['n_mse']  = normalize(results_df['mse'], reverse=True)
    results_df['n_ssim'] = normalize(results_df['ssim'], reverse=False)

    # --- SUCCESS SCORE BERECHNEN ---
    results_df['success_score'] = (results_df['n_mae'] + results_df['n_mse'] + results_df['n_ssim']) / 3

    # Ranking erstellen: Beste Scores oben
    results_df = results_df.sort_values('success_score', ascending=False).reset_index(drop=True)

    # --- OUTPUT IN TXT DATEI ---
    output_file = base_path / OUTPUT_FILENAME
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*160 + "\n")
        f.write(f"MASTER THESIS EVALUATION - RERUN TRIPLE LOSS\n")
        f.write(f"Ranked by Normalized Success Score (NCS) | Scaling: MSE*100, MAE*10\n")
        f.write("="*160 + "\n\n")
        
        header = f"{'Rank':<5} {'Run Name':<70} {'Source':<30} {'Score':<8} {'MSE*100':<10} {'SSIM':<8} {'MAE*10':<10} {'Epoch':<5}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for i, row in results_df.iterrows():
            display_mse = row['mse'] * 100
            display_mae = row['mae'] * 10
            
            line = (f"{i+1:<5} {row['run_name'][:68]:<70} {row['origin'][:28]:<30} {row['success_score']:8.4f} "
                    f"{display_mse:10.6f} {row['ssim']:8.4f} {display_mae:10.6f} {int(row['best_epoch']):<5}\n")
            f.write(line)

    print(f"Evaluation abgeschlossen! {len(results_df)} Runs analysiert.")
    print(f"Ergebnisse gespeichert in: {output_file}")

if __name__ == "__main__":
    evaluate_all_runs(BASE_DIR)