import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
# Wir zielen jetzt direkt auf den Unterordner mit allen CSVs
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV\csv_triple_loss") 

def evaluate_all_runs(base_path):
    all_results = []

    # Suche alle CSV Dateien rekursiv innerhalb von 'csv_triple_loss'
    csv_files = list(base_path.glob("**/log_*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {base_path} gefunden! Prüfe den Pfad.")
        return

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # 1. Finde die Zeile mit dem NIEDRIGSTEN val_loss (Best-Point)
            if 'val_loss' not in df.columns:
                continue
                
            best_idx = df['val_loss'].idxmin()
            best_row = df.loc[best_idx]
            
            # Daten extrahieren
            run_name = csv_path.stem.replace("log_", "")
            all_results.append({
                'run_name': run_name,
                'best_epoch': int(best_row['epoch']),
                'val_loss': best_row['val_loss'],
                'mae': best_row['val_mae_center'],
                'mse': best_row['val_mse_center'],
                'ssim': best_row['val_ssim_center'],
                'psnr': best_row['val_psnr_center']
            })
        except Exception as e:
            print(f"Fehler beim Lesen von {csv_path.name}: {e}")

    # In DataFrame umwandeln
    results_df = pd.DataFrame(all_results)

    # 2. NORMALISIERUNG (NCS Berechnung)
    def normalize(series, reverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min: return series * 0 + 1.0
        if reverse: # Kleiner ist besser (MAE, MSE)
            return (s_max - series) / (s_max - s_min)
        else: # Größer ist besser (SSIM)
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
        f.write("="*120 + "\n")
        f.write(f"MASTER THESIS EVALUATION - FOLDER: {base_path.name}\n")
        f.write(f"Ranked by Normalized Composite Score (NCS) at Min Val_Loss point\n")
        f.write("="*120 + "\n\n")
        
        header = f"{'Rank':<5} {'Run Name':<60} {'Score':<8} {'PSNR':<8} {'SSIM':<8} {'MAE':<10} {'Epoch':<5}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for i, row in results_df.iterrows():
            line = (f"{i+1:<5} {row['run_name'][:58]:<60} {row['success_score']:8.4f} "
                    f"{row['psnr']:8.2f} {row['ssim']:8.4f} {row['mae']:10.6f} {int(row['best_epoch']):<5}\n")
            f.write(line)

    print(f"Evaluation abgeschlossen! {len(results_df)} Runs im Ordner '{base_path.name}' analysiert.")
    print(f"Zusammenfassung gespeichert in: {output_file}")

if __name__ == "__main__":
    evaluate_all_runs(BASE_DIR)