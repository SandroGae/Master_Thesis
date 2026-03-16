#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
CARE_DIR = BASE_DIR / "npz_files" / "CARE_10_SEEDS"
MIXED_DIR = BASE_DIR / "npz_files" / "Best_3_Points"
OUT_DIR = BASE_DIR / "Thesis_Quantitative_Plots" / "Model_Comparison"

OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'grid.alpha': 0.3
})

MODELS = {
    'CARE (Baseline)': (CARE_DIR, 'Confidence_MSE', '#7f8c8d'),
    'Modell P02': (MIXED_DIR, 'P02', '#1f77b4'),
    'Modell P14': (MIXED_DIR, 'P14', '#ff7f0e'),
    'Modell P23': (MIXED_DIR, 'P23', '#2ca02c')
}

# =====================================================
# 2. DATEN LADE-FUNKTION (Mit 3D Fix)
# =====================================================
def load_model_data(name, folder, prefix):
    print(f"\n-> Lade Daten für {name}...")
    
    if not folder.exists():
        print(f"  [Warnung] Ordner {folder} nicht gefunden.")
        return None

    all_npzs = sorted(list(folder.glob(f"*{prefix}*.npz")))
    series_dict = defaultdict(list)
    
    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            series_dict[s_id].append(f)
            
    if not series_dict:
        print(f"  [Warnung] Keine Dateien für {name} gefunden.")
        return None

    global_err, global_alea, global_epis = [], [], []
    ssim_scores, mae_scores = [], []
    
    for s_id, files in tqdm(series_dict.items(), desc=f"Verarbeite {name}"):
        if len(files) != 10: continue
        
        mus, sigmas = [], []
        for idx, path in enumerate(files):
            data = np.load(path)
            mus.append(data['pred'][2:-2, :, :])
            sigmas.append(data['sigma'][2:-2, :, :])
            if idx == 0: gt = data['gt'][2:-2, :, :]
                
        mu_ens = np.mean(mus, axis=0)
        sigma_alea = np.sqrt(np.mean(np.array(sigmas)**2, axis=0))
        sigma_epis = np.std(mus, axis=0)
        
        global_err.append(np.abs(mu_ens - gt).flatten())
        global_alea.append(sigma_alea.flatten())
        global_epis.append(sigma_epis.flatten())
        
        for i in range(mu_ens.shape[0]):
            slice_gt = gt[i]
            slice_mu = mu_ens[i]
            mae_scores.append(np.mean(np.abs(slice_mu - slice_gt)))
            
            dr = max(slice_gt.max() - slice_gt.min(), 1e-5)
            s = ssim(slice_gt, slice_mu, data_range=dr)
            ssim_scores.append(s)

    return {
        'error': np.concatenate(global_err),
        'alea': np.concatenate(global_alea),
        'epis': np.concatenate(global_epis),
        'ssim': ssim_scores,
        'mae': mae_scores
    }

# =====================================================
# 3. MASTER PLOT GENERATOR
# =====================================================
def create_master_dashboard(data_dict):
    print("\n>>> Generiere 2x2 Master-Dashboard mit Kurven und Errorbars...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    valid_models = {k: v for k, v in data_dict.items() if v is not None}
    names = list(valid_models.keys())
    colors = [MODELS[name][2] for name in names]

    # ---------------------------------------------------------
    # Quadrant 1 (Oben Links): SSIM Violin Plot (Unverändert gut)
    # ---------------------------------------------------------
    ax = axes[0, 0]
    ssim_data = [valid_models[n]['ssim'] for n in names]
    
    parts = ax.violinplot(ssim_data, showmeans=True, showextrema=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_title("Structural Similarity Index (SSIM) pro Slice\n(Höher = Besser)")
    ax.set_ylabel("SSIM Score")
    ax.grid(True)

    # ---------------------------------------------------------
    # Quadrant 2 (Oben Rechts): Smoothed Error Distribution
    # ---------------------------------------------------------
    ax = axes[0, 1]
    for name in names:
        err = valid_models[name]['error']
        
        # Berechne das zackige Histogramm
        counts, bin_edges = np.histogram(err, bins=200, range=(0, np.percentile(err, 99)), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # NEU: Gauß-Filter für weiche, fließende Kurven
        smoothed_counts = gaussian_filter1d(counts, sigma=4)
        
        ax.plot(bin_centers, smoothed_counts, linewidth=2.5, color=MODELS[name][2], label=name)
        
    ax.set_yscale('log')
    ax.set_title("Absolute Error Distribution (Smoothed)\n(Weiter links/Steiler abfallend = Besser)")
    ax.set_xlabel(r"Absoluter Fehler ($|Pred - GT|$)")
    ax.set_ylabel("Density (Log Scale)")
    ax.legend()
    ax.grid(True)

    # ---------------------------------------------------------
    # Quadrant 3 (Unten Links): Calibration mit Kurven & Errorbars
    # ---------------------------------------------------------
    ax = axes[1, 0]
    max_sigma = max([np.percentile(valid_models[n]['alea'], 99.0) for n in names])
    ax.plot([0, max_sigma], [0, max_sigma], 'k--', alpha=0.6, label="Ideal Calibration")
    
    num_bins = 15 # Etwas weniger Bins machen Errorbars übersichtlicher
    for name in names:
        alea = valid_models[name]['alea']
        err = valid_models[name]['error']
        bins = np.linspace(0, max_sigma, num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        obs_rmse, obs_std, valid_centers = [], [], []
        for i in range(len(bins)-1):
            mask = (alea >= bins[i]) & (alea < bins[i+1])
            if np.sum(mask) > 1000:
                bin_errs = err[mask]
                obs_rmse.append(np.sqrt(np.mean(bin_errs**2)))
                obs_std.append(np.std(bin_errs)) # Standardabweichung als Errorbar
                valid_centers.append(bin_centers[i])
                
        valid_centers = np.array(valid_centers)
        obs_rmse = np.array(obs_rmse)
        obs_std = np.array(obs_std)

        # Smooth Spline Interpolation (Kurve fitten)
        if len(valid_centers) > 3:
            spline = make_interp_spline(valid_centers, obs_rmse, k=2) 
            x_smooth = np.linspace(valid_centers.min(), valid_centers.max(), 200)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=MODELS[name][2], alpha=0.8, linewidth=2.0)

        # NEU: Angepasstes Errorbar-Design (Konsistent mit Profil-Code)
        ax.errorbar(valid_centers, obs_rmse, yerr=obs_std, fmt='o', markersize=5, 
                    elinewidth=1.0, capsize=2.0, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Reliability Diagram (Aleatoric Calibration)\n(Näher an gestrichelter Linie = Besser)")
    ax.set_xlabel(r"Predicted Aleatoric Uncertainty ($\sigma_{alea}$)")
    ax.set_ylabel("Observed Error (RMSE) $\pm$ Std")
    ax.legend()
    ax.grid(True)

    # ---------------------------------------------------------
    # Quadrant 4 (Unten Rechts): Sparsification mit Kurven & Errorbars
    # ---------------------------------------------------------
    ax = axes[1, 1]
    fractions = np.linspace(0, 0.95, 15) 
    
    for name in names:
        err = valid_models[name]['error']
        epis = valid_models[name]['epis']
        
        sorted_indices = np.argsort(epis)[::-1]
        sorted_errors = err[sorted_indices]
        
        rem_rmse, rem_std = [], []
        for f in fractions:
            cutoff = int(len(sorted_errors) * f)
            rem_errs = sorted_errors[cutoff:]
            rem_rmse.append(np.sqrt(np.mean(rem_errs**2)))
            rem_std.append(np.std(rem_errs))
            
        x_val = fractions * 100
        y_val = np.array(rem_rmse)
        y_err = np.array(rem_std)

        # Smooth Spline Interpolation
        if len(x_val) > 3:
            spline = make_interp_spline(x_val, y_val, k=3) 
            x_smooth = np.linspace(x_val.min(), x_val.max(), 200)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=MODELS[name][2], alpha=0.8, linewidth=2.0)

        # NEU: Angepasstes Errorbar-Design (Konsistent mit Profil-Code)
        ax.errorbar(x_val, y_val, yerr=y_err, fmt='o', markersize=5, 
                    elinewidth=1.0, capsize=2.0, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Sparsification Curve (Epistemic)\n(Tieferer Drop = Besser)")
    ax.set_xlabel("Pixels Removed (Highest Uncertainty First) [%]")
    ax.set_ylabel("RMSE of Remaining Pixels $\pm$ Std")
    ax.legend()
    ax.grid(True)

    # Finish & Save
    plt.tight_layout(pad=3.0)
    save_path = OUT_DIR / "Master_Comparison_Dashboard_Smooth.png"
    plt.savefig(save_path)
    print(f"✅ Hochwertiges Dashboard gespeichert unter: {save_path}")
    plt.close()

if __name__ == "__main__":
    data_dict = {}
    for name, (folder, prefix, _) in MODELS.items():
        data_dict[name] = load_model_data(name, folder, prefix)
        
    create_master_dashboard(data_dict)