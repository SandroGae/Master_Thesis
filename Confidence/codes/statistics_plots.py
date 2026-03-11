#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
EXP_NAME = "CARE_10_SEEDS"
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR = BASE_DIR / "npz_files" / EXP_NAME
OUT_DIR = BASE_DIR / "Thesis_Quantitative_Plots" / EXP_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Professionelles Thesis-Styling
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

# =====================================================
# 2. DATEN AGGREGATION (Globale Analyse)
# =====================================================
def run_global_quantitative_analysis():
    print(">>> Sammle Daten aus ALLEN verfügbaren Serien...")
    
    all_npzs = sorted(list(NPZ_DIR.glob("*.npz")))
    ensembles = defaultdict(list)

    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            ensembles[s_id].append(f)

    global_mu, global_gt, global_alea, global_epis = [], [], [], []
    valid_series_count = 0

    for s_id, file_paths in tqdm(ensembles.items(), desc="Verarbeite Serien"):
        if len(file_paths) != 10:
            continue
            
        valid_series_count += 1
        mus, sigmas = [], []
        gt = None
        
        for idx, path in enumerate(file_paths):
            data = np.load(path)
            mus.append(data['pred'])
            sigmas.append(data['sigma'])
            if idx == 0: gt = data['gt']
                
        # Slicing (Ränder weg)
        mus = np.stack(mus)[:, 2:-2, :, :]
        sigmas = np.stack(sigmas)[:, 2:-2, :, :]
        gt = gt[2:-2, :, :]
        
        # Ensemble Metriken
        mu_ens = np.mean(mus, axis=0)
        sigma_alea = np.sqrt(np.mean(sigmas**2, axis=0))
        sigma_epis = np.std(mus, axis=0)
        
        global_mu.append(mu_ens.flatten())
        global_gt.append(gt.flatten())
        global_alea.append(sigma_alea.flatten())
        global_epis.append(sigma_epis.flatten())

    print(f"\n>>> Daten von {valid_series_count} Serien erfolgreich aggregiert.")
    
    global_mu = np.concatenate(global_mu)
    global_gt = np.concatenate(global_gt)
    global_alea = np.concatenate(global_alea)
    global_epis = np.concatenate(global_epis)
    
    global_error = np.abs(global_mu - global_gt)
    global_rmse_baseline = np.sqrt(np.mean(global_error**2))
    
    print(f"-> Analysiere {len(global_mu):,} Pixel für die Thesis-Plots...\n")

    # =========================================================
    # PLOT 1: SPARSIFICATION CURVE (Fokus: Epistemisch) -> Fancy
    # =========================================================
    print("-> Erstelle Sparsification Curve (Fancy Style)...")
    sorted_indices = np.argsort(global_epis)[::-1]
    sorted_errors = global_error[sorted_indices]
    
    fractions = np.linspace(0, 0.95, 20)
    remaining_rmse = []
    
    for f in fractions:
        cutoff = int(len(sorted_errors) * f)
        rmse = np.sqrt(np.mean(sorted_errors[cutoff:]**2))
        remaining_rmse.append(rmse)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(y=global_rmse_baseline, color='#7f8c8d', linestyle='--', label="Random Removal (Baseline)")
    ax.plot(fractions * 100, remaining_rmse, marker='o', color='#1f77b4', label="Ensemble Disagreement")
    ax.fill_between(fractions * 100, remaining_rmse, global_rmse_baseline, color='#1f77b4', alpha=0.1)
    
    ax.set_title("Sparsification: Model Disagreement vs. Error")
    ax.set_xlabel("Pixels Removed (Highest Uncertainty First) [%]")
    ax.set_ylabel("RMSE of Remaining Pixels")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Plot_1_Sparsification_Fancy.png")
    plt.close()

    # =========================================================
    # PLOT 2: CALIBRATION PLOT (Original Integrated Style) -> FIXED
    # =========================================================
    print("-> Erstelle Calibration Plot (Original Integrated Style)...")
    num_bins = 20
    max_sigma = np.percentile(global_alea, 99.0)
    bins = np.linspace(global_alea.min(), max_sigma, num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    observed_rmse, valid_centers, pixel_counts = [], [], []
    for i in range(len(bins)-1):
        mask = (global_alea >= bins[i]) & (global_alea < bins[i+1])
        count = np.sum(mask)
        if count > 1000:
            rmse = np.sqrt(np.mean(global_error[mask]**2))
            observed_rmse.append(rmse)
            valid_centers.append(bin_centers[i])
            pixel_counts.append(count)

    # Replikation von Image 7ae8fb.jpg
    fig, ax = plt.subplots(figsize=(9, 9))
    ax_bars = ax.twinx() # Erzeuge Twin-Achse für das Histogramm

    # 1. Bar chart im Hintergrund auf Twin-Achse
    ax_bars.bar(valid_centers, pixel_counts, width=(bins[1]-bins[0])*0.8, color='gray', alpha=0.5)
    ax_bars.set_yscale('log') # Log-Skala für Counts (wie im Original)
    ax_bars.set_ylabel("Pixel Count (Log)", fontsize=14, color='gray')
    ax_bars.tick_params(axis='y', labelcolor='gray')

    # 2. Main plots im Vordergrund auf Haupt-Achse (ax)
    ax.plot([0, max_sigma], [0, max_sigma], 'k--', alpha=0.9, label="Perfect Calibration")
    ax.plot(valid_centers, observed_rmse, marker='s', color='darkorange', label="Model Calibration")
    
    ax.set_title("Reliability Diagram (Aleatoric Calibration)")
    ax.set_xlabel(r"Predicted Aleatoric Uncertainty ($\sigma_{alea}$)")
    ax.set_ylabel("Observed Error (RMSE)")
    ax.grid(True, alpha=0.3)
    
    # Integrierte Legende (muss von beiden Achsen gesammelt werden)
    import matplotlib.patches as mpatches
    h1, l1 = ax.get_legend_handles_labels()
    # Proxy für das Balkendiagramm-Label
    h2 = [mpatches.Patch(color='gray', alpha=0.5)]
    l2 = ['Pixel Count']
    ax.legend(h1+h2, l1+l2, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Plot_2_Calibration_OriginalIntegrated.png")
    plt.close()

    # =========================================================
    # PLOT 3: SATURATION ANALYSIS (GT > 1.0) -> Fancy
    # =========================================================
    print("-> Erstelle Saturation Analysis (Fancy Style)...")
    mask_sat = global_gt > 1.0
    mask_reg = global_gt <= 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(r"Impact of Intensity Saturation (Ground Truth $> 1.0$)", fontsize=20)
    
    # Aleatoric
    axes[0].hist(global_alea[mask_reg], bins=100, density=True, alpha=0.5, color='#7f8c8d', 
                 label=r'Normal (GT $\leq$ 1.0)', range=(0, max_sigma), edgecolor='black', linewidth=0.5)
    axes[0].hist(global_alea[mask_sat], bins=100, density=True, alpha=0.7, color='#c0392b', 
                 label=r'Saturated (GT $>$ 1.0)', range=(0, max_sigma), edgecolor='black', linewidth=0.5)
    axes[0].set_title(r"Aleatoric Shift ($\sigma_{alea}$)")
    axes[0].legend()

    # Epistemic
    max_epis = np.percentile(global_epis, 99.5)
    axes[1].hist(global_epis[mask_reg], bins=100, density=True, alpha=0.5, color='#7f8c8d', 
                 label=r'Normal (GT $\leq$ 1.0)', range=(0, max_epis), edgecolor='black', linewidth=0.5)
    axes[1].hist(global_epis[mask_sat], bins=100, density=True, alpha=0.7, color='#2980b9', 
                 label=r'Saturated (GT $>$ 1.0)', range=(0, max_epis), edgecolor='black', linewidth=0.5)
    axes[1].set_title("Epistemic Shift (Disagreement)")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("Predicted Uncertainty")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Plot_3_Saturation_Fancy.png")
    plt.close()

if __name__ == "__main__":
    run_global_quantitative_analysis()
    print(">>> Alle fancy Master-Thesis Plots wurden erfolgreich erstellt!")