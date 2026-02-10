import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

# Basispfade
PARENT_CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV")
FIG_BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Figures")

def get_selection(items, prompt, multi_select=True):
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"[{i}] {item}")
    
    while True:
        try:
            selection = input("\nAuswahl (z.B. '0,2,5-10' oder '1-23'): ")
            parts = selection.split(",")
            indices = []
            
            for part in parts:
                part = part.strip()
                if "-" in part:
                    # Verarbeitet Spannen wie 1-23
                    start, end = part.split("-")
                    indices.extend(range(int(start), int(end) + 1))
                else:
                    # Verarbeitet einzelne Zahlen
                    indices.append(int(part))
            
            # Duplikate entfernen und sortieren
            indices = sorted(list(set(indices)))

            if all(0 <= i < len(items) for i in indices):
                if not multi_select and len(indices) > 1:
                    print("Bitte nur eine Nummer wählen.")
                    continue
                return [items[i] for i in indices]
            else:
                print(f"Zahlen zwischen 0 und {len(items)-1} wählen.")
        except ValueError:
            print("Ungültige Eingabe. Bitte Format wie '1,2,5' oder '1-5' verwenden.")

def main():
    # 1. Ordner-Auswahl
    try:
        subdirs = sorted([d.name for d in PARENT_CSV_DIR.iterdir() if d.is_dir()])
    except OSError: return

    if not subdirs: return
    selected_folder = get_selection(subdirs, "Welchen Ordner möchtest du öffnen?", multi_select=False)[0]
    
    current_csv_dir = PARENT_CSV_DIR / selected_folder
    current_fig_dir = FIG_BASE_DIR / selected_folder
    current_fig_dir.mkdir(parents=True, exist_ok=True)

    # 2. Datei-Auswahl
    all_files = sorted([f.name for f in current_csv_dir.glob("*.csv")])
    if not all_files: return
    selected_files = get_selection(all_files, f"Welche Dateien aus '{selected_folder}' plotten?")

    # 3. Metriken-Auswahl
    sample_df = pd.read_csv(current_csv_dir / selected_files[0])
    all_metrics = [c for c in sample_df.columns if c not in ['epoch', 'lr', 'Unnamed: 0']]
    selected_metrics = get_selection(all_metrics, "Welche Metriken?")

    # 4. Plotting
    for metric in selected_metrics:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        val_list = []
        labels_val = []
        is_error = any(m in metric.lower() for m in ["loss", "mae", "mse"])

        for file in selected_files:
            try:
                df = pd.read_csv(current_csv_dir / file)
                label = file.replace(".csv", "").replace("log_", "")
                x_data = df['epoch'] if 'epoch' in df.columns else df.index
                y_data = df[metric]
                
                ax.plot(x_data, y_data, label=label, linewidth=1.5, alpha=0.8)
                
                # Statistik-Wert sammeln
                extrema = y_data.min() if is_error else y_data.max()
                val_list.append(extrema)
                prefix = "Min" if is_error else "Max"
                labels_val.append(f"{prefix} ({label}): {extrema:.4f}")
            except Exception as e: print(f"Fehler bei {file}: {e}")

        # --- LEGENDE ---
        leg1 = ax.legend(loc='upper right', fontsize='x-small', framealpha=0.8)
        ax.add_artist(leg1)

        avg_val = np.mean(val_list)
        std_val = np.std(val_list)
        stats_labels = labels_val + ["---", f"Avg {prefix}: {avg_val:.4f} ± {std_val:.4f}"]
        empty_handles = [Line2D([0], [0], color='none') for _ in stats_labels]

        ax.legend(empty_handles, stats_labels, 
                  loc='upper right', 
                  bbox_to_anchor=(1.0, 0.8), 
                  fontsize='x-small', 
                  title=f"Statistics ({prefix}ima)",
                  title_fontsize='small',
                  framealpha=0.8,
                  handlelength=0, 
                  handletextpad=0)

        ax.set_title(f"Comparison: {metric}\nFolder: {selected_folder}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Speichern
        save_name = f"comparison_{metric}.png"
        plt.savefig(current_fig_dir / save_name, dpi=300)
        print(f"Gespeichert: {save_name}")
        plt.close()

if __name__ == "__main__":
    main()