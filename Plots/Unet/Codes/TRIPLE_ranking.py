import os
import pandas as pd
import numpy as np
from pathlib import Path
from math import log10, floor

# Basispfad
PARENT_CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV")

def round_sig(x, sig=4):
    """Rundet auf eine bestimmte Anzahl signifikanter Stellen."""
    if pd.isna(x) or x == 0:
        return x
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def get_selection(items, prompt, multi_select=True):
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"[{i}] {item}")
    
    while True:
        try:
            selection = input("\nAuswahl (z.B. '1,2,5-10' oder '0'): ")
            parts = selection.split(",")
            indices = []
            for part in parts:
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    indices.extend(range(int(start), int(end) + 1))
                else:
                    indices.append(int(part))
            indices = sorted(list(set(indices)))
            if all(0 <= i < len(items) for i in indices):
                if not multi_select and len(indices) > 1:
                    print("Bitte nur eine Nummer wählen.")
                    continue
                return [items[i] for i in indices]
            else:
                print(f"Zahlen zwischen 0 und {len(items)-1} wählen.")
        except ValueError:
            print("Ungültige Eingabe.")

def main():
    # 1. Ordner-Auswahl
    subdirs = sorted([d.name for d in PARENT_CSV_DIR.iterdir() if d.is_dir()])
    if not subdirs: return
    selected_folder = get_selection(subdirs, "Welchen Ordner analysieren?", multi_select=False)[0]
    current_csv_dir = PARENT_CSV_DIR / selected_folder

    # 2. Datei-Auswahl
    all_files = sorted([f.name for f in current_csv_dir.glob("*.csv") if not f.name.startswith("summary")])
    if not all_files: return
    selected_files = get_selection(all_files, f"Welche Runs aus '{selected_folder}'?")

    # 3. Daten sammeln
    summary_data = []
    for file in selected_files:
        try:
            df = pd.read_csv(current_csv_dir / file)
            # Finde Epoche mit höchster PSNR
            best_idx = df['val_psnr_center'].idxmax()
            best_row = df.loc[best_idx]
            
            run_name = file.replace(".csv", "").replace("log_", "")
            
            # Daten mit signifikanten Stellen runden
            summary_data.append({
                "Run Name": run_name,
                "Epoch": int(best_row['epoch']),
                "PSNR": round_sig(best_row['val_psnr_center'], 4),
                "SSIM": round_sig(best_row['val_ssim_center'], 4),
                "MAE": round_sig(best_row['val_mae_center'], 4),
                "MSE": round_sig(best_row['val_mse_center'], 4)
            })
        except Exception as e:
            print(f"Fehler bei {file}: {e}")

    # 4. Tabelle sortieren
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="PSNR", ascending=False)

        # Tabellen-String generieren
        header = f"ÜBERSICHT: {selected_folder}\nSorted by: PSNR (Best values found per run)\n"
        line = "=" * 110 + "\n"
        table_str = summary_df.to_string(index=False, justify='left', col_space=[40, 7, 10, 10, 10, 10])
        
        final_output = f"{header}{line}{table_str}\n{line}"

        # Im Terminal anzeigen
        print("\n" + final_output)

        # Als TXT speichern
        save_path = current_csv_dir / f"summary_{selected_folder}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        
        print(f"Übersicht gespeichert unter: {save_path}")

if __name__ == "__main__":
    main()