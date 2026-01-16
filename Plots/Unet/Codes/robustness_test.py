import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV\robustness_test")
FIG_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Figures\robustness_test")

FIG_DIR.mkdir(parents=True, exist_ok=True)

def get_selection(items, prompt):
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"[{i}] {item}")
    
    while True:
        try:
            selection = input("\nGib die Nummern ein (getrennt durch Komma): ")
            indices = [int(x.strip()) for x in selection.split(",")]
            if all(0 <= i < len(items) for i in indices):
                return [items[i] for i in indices]
            else:
                print(f"Zahlen zwischen 0 und {len(items)-1} wählen.")
        except ValueError:
            print("Ungültige Eingabe.")

def main():
    try:
        all_files = sorted([f.name for f in CSV_DIR.glob("*.csv")])
    except OSError:
        return

    if not all_files:
        return

    selected_files = get_selection(all_files, "Welche CSV-Dateien möchtest du plotten?")

    try:
        sample_df = pd.read_csv(CSV_DIR / selected_files[0])
        all_metrics = [c for c in sample_df.columns if c not in ['epoch', 'lr', 'Unnamed: 0']]
    except Exception:
        return
    
    selected_metrics = get_selection(all_metrics, "Welche Metriken sollen geplottet werden?")

    for metric in selected_metrics:
        plt.figure(figsize=(10, 6))
        
        types = []
        names_lower = [f.lower() for f in selected_files]
        if any("cross_val" in n for n in names_lower): types.append("cross_validation")
        if any("no_augmentation" in n for n in names_lower): types.append("no_augmentation")
        if any("random_seed" in n for n in names_lower): types.append("random_seeds")
        
        type_str = "_".join(set(types)) if types else "comparison"
        
        interp_suffix = ""
        if any("interpolated" in n for n in names_lower):
            interp_suffix = "_interpolated"

        for file in selected_files:
            try:
                df = pd.read_csv(CSV_DIR / file)
                label = file.replace(".csv", "")
                x_data = df['epoch'] if 'epoch' in df.columns else df.index
                plt.plot(x_data, df[metric], label=label, linewidth=1.5, alpha=0.8)
            except Exception:
                continue

        plt.title(f"Robustness Test: {metric}", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(loc='upper right', fontsize='small', framealpha=0.8)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_name = f"{type_str}{interp_suffix}_{metric}.png"
        save_path = FIG_DIR / save_name
        
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Gespeichert: {save_name}")
        except OSError:
            pass
        finally:
            plt.close()

if __name__ == "__main__":
    main()