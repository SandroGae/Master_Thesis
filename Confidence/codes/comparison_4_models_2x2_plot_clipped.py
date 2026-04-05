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
from scipy.stats import norm
import matplotlib.patches as mpatches 
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
# 2. DATEN LADE-FUNKTION (Mit Clipping)
# =====================================================
def load_model_data(name, folder, prefix):
    print(f"\n-> Lade Daten für {name}...")
    if not folder.exists():
        print(f"  [Warnung] Ordner {folder} nicht gefunden."); return None

    all_npzs = sorted(list(folder.glob(f"*{prefix}*.npz")))
    series_dict = defaultdict(list)
    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match: series_dict[int(match.group(1))].append(f)
            
    if not series_dict: return None

    global_err, global_raw_err, global_alea, global_epis = [], [], [], []
    ssim_scores = []
    
    for s_id, files in tqdm(series_dict.items(), desc=f"Verarbeite {name}"):
        if len(files) != 10: continue
        mus, sigmas = [], []
        for idx, path in enumerate(files):
            data = np.load(path)
            
            # NEU: Vorhersagen auf das physikalische Limit des Netzwerks [0, 1] clippen
            pred_clipped = np.clip(data['pred'][2:-2, :, :], 0.0, 1.0)
            mus.append(pred_clipped)
            
            sigmas.append(data['sigma'][2:-2, :, :])
            
            if idx == 0: 
                # NEU: Ground Truth ebenfalls auf [0, 1] clippen für faire Fehlermessung
                gt = np.clip(data['gt'][2:-2, :, :], 0.0, 1.0)
                
        mu_ens = np.mean(mus, axis=0)
        sigma_alea = np.sqrt(np.mean(np.array(sigmas)**2, axis=0))
        sigma_epis = np.std(mus, axis=0)
        
        global_err.append(np.abs(mu_ens - gt).flatten())
        global_raw_err.append((gt - mu_ens).flatten()) 
        global_alea.append(sigma_alea.flatten())
        global_epis.append(sigma_epis.flatten())
        
        for i in range(mu_ens.shape[0]):
            dr = max(gt[i].max() - gt[i].min(), 1e-5)
            ssim_scores.append(ssim(gt[i], mu_ens[i], data_range=dr))

    return {
        'error': np.concatenate(global_err),
        'raw_err': np.concatenate(global_raw_err),
        'alea': np.concatenate(global_alea),
        'epis': np.concatenate(global_epis),
        'ssim': ssim_scores
    }

# =====================================================
# 3. SPLIT DASHBOARD GENERATOR
# =====================================================
def create_split_dashboards(data_dict):
    valid_models = {k: v for k, v in data_dict.items() if v is not None}
    names = list(valid_models.keys())
    colors = [MODELS[name][2] for name in names]

    # =========================================================
    # FIGURE 1: MU-CHANNEL EVALUATION (1x3)
    # =========================================================
    print("\n>>> Generiere Mu-Channel Dashboard (1x3)...")
    fig_mu, axes_mu = plt.subplots(1, 3, figsize=(24, 7))
    
    ax = axes_mu[0]
    parts = ax.violinplot([valid_models[n]['ssim'] for n in names], showmeans=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    ax.set_xticks(np.arange(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_title("SSIM pro Slice\n(Höher = Besser)"); ax.set_ylabel("SSIM Score"); ax.grid(True)

    ax = axes_mu[1]
    for name in names:
        err = valid_models[name]['error']
        counts, bin_edges = np.histogram(err, bins=200, range=(0, np.percentile(err, 99)), density=True)
        ax.plot((bin_edges[:-1] + bin_edges[1:])/2, gaussian_filter1d(counts, sigma=4), color=MODELS[name][2], lw=2.5, label=name)
    ax.set_yscale('log'); ax.set_title("Absolute Error Distribution\n(Weiter links/Steiler abfallend = Besser)"); ax.legend(); ax.grid(True)
    ax.set_xlabel(r"Absoluter Fehler ($|Pred - GT|$)"); ax.set_ylabel("Density (Log Scale)")

    ax = axes_mu[2]
    fractions = np.linspace(0, 0.95, 15)
    for name in names:
        err, epis = valid_models[name]['error'], valid_models[name]['epis']
        sorted_err = err[np.argsort(epis)[::-1]]
        rem_rmse, rem_sem = [], []
        for f in fractions:
            r_err = sorted_err[int(len(sorted_err) * f):]
            rem_rmse.append(np.sqrt(np.mean(r_err**2)))
            rem_sem.append(np.std(r_err) / np.sqrt(len(r_err))) 
            
        x, y, yerr = fractions * 100, np.array(rem_rmse), np.array(rem_sem)
        if len(x) > 3:
            spline = make_interp_spline(x, y, k=3)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, spline(xs), color=MODELS[name][2], alpha=0.8, lw=2.0)
        ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=5, capsize=2, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Sparsification Curve (Epistemic)\n(Tieferer Drop = Besser)")
    ax.set_xlabel("Pixels Removed (Highest Uncertainty First) [%]")
    ax.set_ylabel("RMSE of Remaining Pixels $\pm$ SEM")
    ax.legend(); ax.grid(True)

    plt.tight_layout(pad=3.0)
    save_path_mu = OUT_DIR / "Mu_Channel_Evaluation_Clipped.png"
    plt.savefig(save_path_mu); plt.close()
    print(f"✅ Mu-Dashboard gespeichert: {save_path_mu}")

    # =========================================================
    # FIGURE 2: SIGMA-CHANNEL EVALUATION (Jetzt 1x2)
    # =========================================================
    print("\n>>> Generiere Sigma-Channel Dashboard (1x2)...")
    # Änderung: Nur noch 2 Spalten, Breite auf 16 reduziert
    fig_sig, axes_sig = plt.subplots(1, 2, figsize=(16, 8)) 
    fig_sig.suptitle("Uncertainty Quantification & Calibration ($\sigma$-Channel)", fontsize=22, fontweight='bold', y=1.05)
    
    # 1. Reliability Diagram
    ax = axes_sig[0]; ax_hist = ax.twinx()
    max_sigma = max([np.percentile(valid_models[n]['alea'], 99.0) for n in names])
    num_bins = 15
    bins = np.linspace(0, max_sigma, num_bins)
    
    all_alea = np.concatenate([valid_models[n]['alea'] for n in names])
    cnt, _ = np.histogram(all_alea, bins=bins)
    ax_hist.bar((bins[:-1] + bins[1:])/2, cnt, width=(bins[1]-bins[0])*0.8, color='gray', alpha=0.15)
    ax_hist.set_yscale('log'); ax_hist.set_ylim(1, cnt.max() * 2.0)
    ax_hist.set_ylabel("Sample Count (Log)", color='gray')

    ax.plot([0, max_sigma], [0, max_sigma], 'k--', alpha=0.6, label="Ideal Calibration")
    
    for name in names:
        alea, err = valid_models[name]['alea'], valid_models[name]['error']
        obs_rmse, obs_sem, valid_centers = [], [], []
        for i in range(len(bins)-1):
            mask = (alea >= bins[i]) & (alea < bins[i+1])
            if np.sum(mask) > 1000:
                b_err = err[mask]
                obs_rmse.append(np.sqrt(np.mean(b_err**2)))
                obs_sem.append(np.std(b_err) / np.sqrt(np.sum(mask)))
                valid_centers.append((bins[i] + bins[i+1])/2)
        
        ax.errorbar(valid_centers, obs_rmse, yerr=obs_sem, fmt='o', markersize=5, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Reliability Diagram (Aleatoric Calibration)")
    ax.set_xlabel(r"Predicted Aleatoric Uncertainty ($\sigma_{alea}$)")
    ax.set_ylabel("Observed Error (RMSE)")
    ax.legend(loc='upper left'); ax.grid(True)

    # 2. Z-Score Distribution
    ax = axes_sig[1]
    x_ideal = np.linspace(-5, 5, 200)
    
    # Verbesserung: Höhere Linienstärke (lw=4) und zorder=10, damit sie oben liegt
    ax.plot(x_ideal, norm.pdf(x_ideal, 0, 1), 'k--', lw=4, alpha=1.0, 
            label="Ideal Unit Normal $N(0,1)$", zorder=10)
    
    for name in names:
        raw_err = valid_models[name]['raw_err']
        alea = valid_models[name]['alea']
        z_scores = raw_err / (alea + 1e-8) 
        counts, bin_edges = np.histogram(z_scores, bins=250, range=(-5, 5), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        ax.plot(bin_centers, gaussian_filter1d(counts, sigma=4), linewidth=2.5, 
                color=MODELS[name][2], alpha=0.8, label=name)
        
    ax.set_title("Standardized Residuals (Z-Score)")
    ax.set_xlabel(r"Z-Score $(GT - \mu_{ens}) / \sigma_{alea}$")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right'); ax.grid(True)

    # Der Block für axes_sig[2] (ECE) wurde komplett entfernt

    plt.tight_layout(pad=3.0)
    save_path_sig = OUT_DIR / "Sigma_Channel_Evaluation_Final.png"
    plt.savefig(save_path_sig, bbox_inches='tight'); plt.close()
    print(f"✅ Sigma-Dashboard gespeichert: {save_path_sig}")

if __name__ == "__main__":
    data_dict = {name: load_model_data(name, f, p) for name, (f, p, c) in MODELS.items()}
    create_split_dashboards(data_dict)