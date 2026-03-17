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
# 2. DATEN LADE-FUNKTION
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

    global_err, global_alea, global_epis = [], [], []
    ssim_scores = []
    
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
            dr = max(gt[i].max() - gt[i].min(), 1e-5)
            ssim_scores.append(ssim(gt[i], mu_ens[i], data_range=dr))

    return {
        'error': np.concatenate(global_err),
        'alea': np.concatenate(global_alea),
        'epis': np.concatenate(global_epis),
        'ssim': ssim_scores
    }

# =====================================================
# 3. MASTER PLOT GENERATOR
# =====================================================
def create_master_dashboard(data_dict):
    print("\n>>> Generiere 2x2 Master-Dashboard mit SEM-Fehlerbalken...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    valid_models = {k: v for k, v in data_dict.items() if v is not None}
    names = list(valid_models.keys())
    colors = [MODELS[name][2] for name in names]

    # Q1: SSIM Violin
    ax = axes[0, 0]
    parts = ax.violinplot([valid_models[n]['ssim'] for n in names], showmeans=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    ax.set_xticks(np.arange(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_title("SSIM pro Slice\n(Höher = Besser)"); ax.set_ylabel("SSIM Score"); ax.grid(True)

    # Q2: Error Dist
    ax = axes[0, 1]
    for name in names:
        err = valid_models[name]['error']
        counts, bin_edges = np.histogram(err, bins=200, range=(0, np.percentile(err, 99)), density=True)
        ax.plot((bin_edges[:-1] + bin_edges[1:])/2, gaussian_filter1d(counts, sigma=4), color=MODELS[name][2], label=name)
    ax.set_yscale('log'); ax.set_title("Absolute Error Distribution"); ax.legend(); ax.grid(True)

    # Q3: Calibration (FIXED mit SEM)
    ax = axes[1, 0]; ax_hist = ax.twinx()
    max_sigma = max([np.percentile(valid_models[n]['alea'], 99.0) for n in names])
    num_bins = 15
    bins = np.linspace(0, max_sigma, num_bins)
    
    # Hintergrund-Histogramm
    all_alea = np.concatenate([valid_models[n]['alea'] for n in names])
    cnt, _ = np.histogram(all_alea, bins=bins)
    ax_hist.bar((bins[:-1] + bins[1:])/2, cnt, width=(bins[1]-bins[0])*0.8, color='gray', alpha=0.15)
    ax_hist.set_yscale('log'); ax_hist.set_ylim(1, cnt.max() * 2.0); ax_hist.tick_params(axis='y', labelcolor='gray')

    ax.plot([0, max_sigma], [0, max_sigma], 'k--', alpha=0.6, label="Ideal Calibration")
    
    for name in names:
        alea, err = valid_models[name]['alea'], valid_models[name]['error']
        obs_rmse, obs_sem, valid_centers = [], [], []
        for i in range(len(bins)-1):
            mask = (alea >= bins[i]) & (alea < bins[i+1])
            n_pix = np.sum(mask)
            if n_pix > 1000:
                b_err = err[mask]
                obs_rmse.append(np.sqrt(np.mean(b_err**2)))
                # SEM BERECHNUNG: Standardabweichung / Wurzel(N)
                obs_sem.append(np.std(b_err) / np.sqrt(n_pix))
                valid_centers.append((bins[i] + bins[i+1])/2)
        
        v_c, o_r, o_s = np.array(valid_centers), np.array(obs_rmse), np.array(obs_sem)
        if len(v_c) > 3:
            spline = make_interp_spline(v_c, o_r, k=2)
            xs = np.linspace(v_c.min(), v_c.max(), 200)
            ax.plot(xs, spline(xs), color=MODELS[name][2], alpha=0.8)
        ax.errorbar(v_c, o_r, yerr=o_s, fmt='o', markersize=5, capsize=2, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Reliability Diagram (SEM-Balken)"); ax.legend(loc='upper left'); ax.grid(True)

    # Q4: Sparsification (FIXED mit SEM)
    ax = axes[1, 1]
    fractions = np.linspace(0, 0.95, 15)
    for name in names:
        err, epis = valid_models[name]['error'], valid_models[name]['epis']
        sorted_err = err[np.argsort(epis)[::-1]]
        rem_rmse, rem_sem = [], []
        for f in fractions:
            r_err = sorted_err[int(len(sorted_err) * f):]
            rem_rmse.append(np.sqrt(np.mean(r_err**2)))
            # SEM BERECHNUNG
            rem_sem.append(np.std(r_err) / np.sqrt(len(r_err)))
            
        x, y, yerr = fractions * 100, np.array(rem_rmse), np.array(rem_sem)
        if len(x) > 3:
            spline = make_interp_spline(x, y, k=3)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, spline(xs), color=MODELS[name][2], alpha=0.8)
        ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=5, capsize=2, color=MODELS[name][2], alpha=0.7, label=name)

    ax.set_title("Sparsification Curve (SEM-Balken)"); ax.legend(); ax.grid(True)

    plt.tight_layout(pad=3.0)
    save_path = OUT_DIR / "Master_Comparison_Dashboard_SEM.png"
    plt.savefig(save_path); plt.close()
    print(f"✅ Dashboard mit SEM-Balken gespeichert: {save_path}")

if __name__ == "__main__":
    data_dict = {name: load_model_data(name, f, p) for name, (f, p, c) in MODELS.items()}
    create_master_dashboard(data_dict)