#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_ROOT = BASE_DIR / "npz_files" / "Best_30_Points"
OUT_DIR = BASE_DIR / "Thesis_Quantitative_Plots" / "Model_Comparison"

OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

# Farben für die Modelle (anpassbar)
COLORS = {'P02': '#1f77b4', 'P14': '#ff7f0e', 'P23': '#2ca02c'}

# =====================================================
# 2. DATEN LADE-FUNKTION
# =====================================================
def load_ensemble_data(point_folder):
    """Lädt alle Serien und aggregiert das 10-Seed-Ensemble für einen Punkt."""
    all_npzs = sorted(list(point_folder.glob("*.npz")))
    series_dict = defaultdict(list)
    
    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            series_dict[s_id].append(f)
            
    global_mu, global_gt, global_alea, global_epis = [], [], [], []
    
    for s_id, files in series_dict.items():
        if len(files) != 10: continue
        
        mus, sigmas = [], []
        for idx, path in enumerate(files):
            data = np.load(path)
            mus.append(data['pred'][:, 2:-2, :, :])
            sigmas.append(data['sigma'][:, 2:-2, :, :])
            if idx == 0: gt = data['gt'][2:-2, :, :]
                
        mu_ens = np.mean(mus, axis=0)
        sigma_alea = np.sqrt(np.mean(np.array(sigmas)**2, axis=0))
        sigma_epis = np.std(mus, axis=0)
        
        global_mu.append(mu_ens.flatten())
        global_gt.append(gt.flatten())
        global_alea.append(sigma_alea.flatten())
        global_epis.append(sigma_epis.flatten())
        
    return {
        'mu': np.concatenate(global_mu),
        'gt': np.concatenate(global_gt),
        'alea': np.concatenate(global_alea),
        'epis': np.concatenate(global_epis)
    }

# =====================================================
# 3. VERGLEICHS-PLOTS
# =====================================================
def plot_combined_calibration(models_data):
    print("\n-> Erstelle Combined Calibration Plot...")
    plt.figure(figsize=(10, 10))
    
    # Finde globales Max-Sigma für die Achsen
    max_sigma = max([np.percentile(data['alea'], 99.0) for data in models_data.values()])
    plt.plot([0, max_sigma], [0, max_sigma], 'k--', alpha=0.6, label="Ideal Calibration (x=y)")
    
    num_bins = 20
    
    for model_name, data in models_data.items():
        alea = data['alea']
        error = np.abs(data['mu'] - data['gt'])
        
        bins = np.linspace(0, max_sigma, num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        obs_rmse, valid_centers = [], []
        for i in range(len(bins)-1):
            mask = (alea >= bins[i]) & (alea < bins[i+1])
            if np.sum(mask) > 1000:
                obs_rmse.append(np.sqrt(np.mean(error[mask]**2)))
                valid_centers.append(bin_centers[i])
                
        plt.plot(valid_centers, obs_rmse, marker='o', color=COLORS.get(model_name, 'black'), 
                 label=f"Modell {model_name}")

    plt.title("Reliability Diagram: Model Comparison")
    plt.xlabel(r"Predicted Aleatoric Uncertainty ($\sigma_{alea}$)")
    plt.ylabel("Observed Error (RMSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Comparison_Calibration.png")
    plt.close()

def plot_spatial_comparison(models_data, series_id=12, slice_id=20, crop_y=(50,150), crop_x=(50,150)):
    # HINWEIS: Für diesen Plot müssen wir die Daten un-flattened laden. 
    # Um das Skript hier nicht zu sprengen, zeige ich dir das Konzept:
    pass

def main():
    print(">>> Starte Modell-Vergleich...")
    model_folders = [d for d in NPZ_ROOT.iterdir() if d.is_dir() and d.name.startswith("P")]
    
    if not model_folders:
        print(f"❌ Keine Ordner (P02, P14 etc.) in {NPZ_ROOT} gefunden.")
        return
        
    models_data = {}
    for folder in model_folders:
        print(f"-> Lade Daten für {folder.name}...")
        models_data[folder.name] = load_ensemble_data(folder)
        
    plot_combined_calibration(models_data)
    print("✅ Vergleichs-Plots erfolgreich erstellt!")

if __name__ == "__main__":
    main()