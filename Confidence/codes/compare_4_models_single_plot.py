#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & CONFIGURATION
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR = BASE_DIR / "npz_files" 
OUT_DIR = BASE_DIR / "Thesis_Plots" / "Combined_Profiles"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_CONFIG = {
    # Block 1
    5:  {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    11: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (40, 140)},
    12: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (27, 127)},
    13: {"slice_idx": 7,  "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (80, 180)},
    15: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (95, 195)},
    16: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (81, 181)},
    17: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (136, 236)},    
    22: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    29: {"slice_idx": 24, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (23, 123)},
    30: {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    # Block 2
    32: {"slice_idx": 31, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (21, 121)},
    35: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (67, 167)},
    36: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (21, 121)},
    38: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (64, 164)},
    41: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (98, 113),  "fit_window": (0, 100)},
    42: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    45: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (41, 141)},
    46: {"slice_idx": 37, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    50: {"slice_idx": 12, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (59, 159)},
    51: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (108, 208)},
    # Block 3
    55: {"slice_idx": 21, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (122, 222)},
    56: {"slice_idx": 9,  "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (106, 206)},
    57: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (136, 236)},
    59: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (137, 237)},
    64: {"slice_idx": 5,  "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (140, 240)},
    67: {"slice_idx": 10, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (0, 100)},
    68: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (5, 105)},
    72: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "fit_window": (4, 104)},
    73: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (42, 142)},
    74: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (100, 115), "fit_window": (32, 132)},
}

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def gaussian_with_offset(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def apply_shift(arr, shift_amount):
    res = np.full(100, np.nan)
    s_st = max(0, -shift_amount)
    s_en = min(100, 100 - shift_amount)
    d_st = max(0, shift_amount)
    d_en = min(100, 100 + shift_amount)
    res[d_st:d_en] = arr[s_st:s_en]
    return res

def fit_profile(x_data, y_data, y_err=None, is_gt=False):
    """Fittet ein Profil. Bei GT ohne y_err, bei Ensemble mit y_err."""
    valid = ~np.isnan(y_data)
    x_v, y_v = x_data[valid], y_data[valid]
    
    if len(y_v) < 10: return None, None
    
    # Initiale Schätzung
    p0 = [np.max(y_v) - np.min(y_v), 50.0, 5.0, np.min(y_v)]
    bounds = ([0, 20, 0.5, -np.inf], [np.inf, 80, 40, np.inf])
    
    sigma_val = None
    if y_err is not None:
        sigma_val = y_err[valid] + 1e-12

    try:
        popt, pcov = curve_fit(gaussian_with_offset, x_v, y_v, p0=p0, sigma=sigma_val, 
                               absolute_sigma=(y_err is not None), bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except:
        return None, None

# =====================================================
# 3. RUNNER
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR.rglob("*.npz"))) 
    models_data = defaultdict(lambda: defaultdict(list))
    
    print("Sammle Dateipfade...")
    for f in all_npzs:
        match = re.search(r"(P\d+|MSE).*_S(\d+)\.npz", f.name)
        if match:
            raw_id = match.group(1)
            p_id = "CARE (MSE)" if raw_id == "MSE" else raw_id
            s_id = int(match.group(2))
            if s_id in SERIES_CONFIG:
                models_data[p_id][s_id].append(f)

    # Speicher für die ausgerichteten Profile
    aligned_profiles = defaultdict(list)
    results = []

    print("Berechne GT-basiertes Alignment in L-Direction...")
    # Wir iterieren über Serien, um das Alignment pro Serie (einmal für alle Modelle) zu machen
    for s_id in SERIES_CONFIG.keys():
        # 1. GT laden und fitten für diese Serie
        # Wir nehmen das GT aus dem ersten verfügbaren Modell-NPZ dieser Serie
        sample_path = None
        for p_id in models_data:
            if s_id in models_data[p_id]:
                sample_path = models_data[p_id][s_id][0]
                break
        
        if sample_path is None: continue
        
        data_gt = np.load(sample_path)
        gt_vol = data_gt['gt'][2:-2, :, :]
        config = SERIES_CONFIG[s_id]
        z, (y_min, y_max) = config["slice_idx"] - 2, config["roi_y"]
        x_min, x_max = config["fit_window"]
        
        gt_profile = np.mean(gt_vol[z, y_min:y_max, x_min:x_max], axis=0)
        x_axis_100 = np.arange(100)
        
        # Gauß-Fit auf GT zur Peak-Bestimmung
        popt_gt, _ = fit_profile(x_axis_100, gt_profile, is_gt=True)
        
        if popt_gt is None:
            # Fallback: Einfaches Argmax falls Fit fehlschlägt
            mu_gt = np.argmax(np.convolve(gt_profile, np.ones(5)/5, mode='same'))
        else:
            mu_gt = popt_gt[1] # Das physikalische Zentrum laut GT-Fit
            
        shift_amount = int(round(50 - mu_gt))
        
        # 2. Alle Modelle für diese Serie laden, sigma_epi berechnen und mit GT-Shift verschieben
        for p_id in models_data:
            if s_id not in models_data[p_id]: continue
            
            file_paths = models_data[p_id][s_id]
            if len(file_paths) != 10: continue
            
            mus = []
            for path in file_paths:
                d = np.load(path)
                mus.append(d['pred'])
            
            mus = np.stack(mus)[:, 2:-2, :, :]
            sigma_epi_vol = np.std(mus, axis=0)
            p_epi = np.mean(sigma_epi_vol[z, y_min:y_max, x_min:x_max], axis=0)
            
            if len(p_epi) == 100:
                aligned_profiles[p_id].append(apply_shift(p_epi, shift_amount))

    # =====================================================
    # 4. PLOTTING & STATISTIK
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    plt.subplots_adjust(top=0.85, wspace=0.25)
    
    x_axis = np.arange(100)
    colors = {"P02": "#1f77b4", "P14": "#ff7f0e", "P23": "#2ca02c", "CARE (MSE)": "#d62728"}

    for p_id, data_list in sorted(aligned_profiles.items()):
        if len(data_list) == 0: continue
            
        stack = np.stack(data_list)
        n_valid = np.sum(~np.isnan(stack), axis=0)
        n_valid = np.where(n_valid == 0, 1, n_valid)
        
        avg_epi = np.nanmean(stack, axis=0)
        err_epi = np.nanstd(stack, axis=0) / np.sqrt(n_valid)
        
        c = colors.get(p_id, "black")
        
        # --- LEFT PLOT ---
        ax1.plot(x_axis, avg_epi, color=c, lw=2.5, label=p_id)
        ax1.fill_between(x_axis, avg_epi - err_epi, avg_epi + err_epi, color=c, alpha=0.15)
        
        # --- RIGHT PLOT ---
        popt, perr = fit_profile(x_axis, avg_epi, err_epi)
        
        if popt is not None:
            A, mu, sig, offset = popt
            err_A, err_offset = perr[0], perr[3]
            cr = (A + offset) / (offset + 1e-9)
            
            fit_curve = gaussian_with_offset(x_axis, *popt)
            ax2.errorbar(x_axis, avg_epi, yerr=err_epi, fmt='o', markersize=4, 
                         elinewidth=1.0, capsize=1.5, color=c, alpha=0.6)
                         
            lbl = f"{p_id} (CR: {cr:.2f}, BG: {offset:.4f})"
            ax2.plot(x_axis, fit_curve, color=c, ls='--', lw=1.5, label=lbl)
            
            results.append({
                "Modell": p_id, 
                "BG Offset (Floor)": f"{offset:.5f} ± {err_offset:.5f}", 
                "Peak Amplitude": f"{A:.5f} ± {err_A:.5f}", 
                "Contrast Ratio": f"{cr:.2f}"
            })
        else:
            ax2.errorbar(x_axis, avg_epi, yerr=err_epi, fmt='o', markersize=4, 
                         elinewidth=1.0, capsize=1.5, color=c, alpha=0.6, label=f"{p_id} (Fit failed)")

    # Styling
    ax1.set_title("Absolute Epistemic Uncertainty (L-Direction)\n(Shaded Standard Error)", fontsize=14, pad=15)
    ax2.set_title("Absolute Epistemic Uncertainty (L-Direction)\n(Scatter Data + Gaussian Fit)", fontsize=14, pad=15)
    
    for ax in [ax1, ax2]:
        ax.set_ylabel("Standard Deviation $\sigma$", fontsize=12)
        ax.set_xlabel("Pixel Index across ROI (GT-Peak aligned at 50)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=11)
        ax.set_xlim(5, 95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Quantitative Uncertainty Analysis: L-Direction (GT-based Alignment)", 
                 fontsize=18, fontweight='bold', y=0.98)

    print("\n--- STATISTIK (GT-ALIGNED L-DIRECTION) ---")
    print(pd.DataFrame(results).to_string(index=False))

    plt.savefig(OUT_DIR / "Final_L-Direction_GT_Aligned.png", bbox_inches='tight')
    plt.show()