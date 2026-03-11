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
# 2. HILFSFUNKTIONEN (Gauss-Fit & dynamisches Alignment)
# =====================================================
def gaussian_with_offset(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def apply_shift(arr, shift_amount):
    """Verschiebt das Array um shift_amount Pixel und füllt Ränder mit NaN."""
    res = np.full(100, np.nan)
    s_st = max(0, -shift_amount)
    s_en = min(100, 100 - shift_amount)
    d_st = max(0, shift_amount)
    d_en = min(100, 100 + shift_amount)
    res[d_st:d_en] = arr[s_st:s_en]
    return res

def fit_ensemble_profile(x_data, y_data, y_err):
    """Fittet die gemittelte Ensemble-Kurve für den rechten Plot."""
    valid = ~np.isnan(y_data)
    x_v, y_v, e_v = x_data[valid], y_data[valid], y_err[valid]
    
    if len(y_v) < 10: return None, None
    
    # Initiale Parameter für ABSOLUTE Uncertainty
    p0 = [np.max(y_v) - np.min(y_v), 50.0, 5.0, np.min(y_v)]
    bounds = ([0, 40, 1, 0], [np.inf, 60, 30, np.inf])
    
    try:
        popt, pcov = curve_fit(gaussian_with_offset, x_v, y_v, p0=p0, sigma=e_v+1e-12, absolute_sigma=True, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except:
        return None, None

# =====================================================
# 3. RUNNER
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR.rglob("*.npz"))) 
    
    # Zuerst alle Dateien sauber nach Modell und Serie sortieren
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

    # Hier speichern wir die 1D-Profile
    raw_profiles = defaultdict(dict)
    
    print("Lade NPZ Dateien und berechne absolute Epistemic Uncertainty...")
    for p_id, series_dict in models_data.items():
        for s_id, file_paths in series_dict.items():
            if len(file_paths) != 10: continue
            
            mus = []
            for path in file_paths:
                data = np.load(path)
                mus.append(data['pred'])
            
            # WICHTIG: Hier stapeln wir alle 10 Seeds! Shape -> (10, Z, Y, X)
            mus = np.stack(mus)[:, 2:-2, :, :]
            
            # Absolute Epistemic Uncertainty berechnen (Standardabweichung über die 10 Seeds)
            sigma_epi = np.std(mus, axis=0) # Shape -> (Z, Y, X)
            
            config = SERIES_CONFIG[s_id]
            z, (y_min, y_max) = config["slice_idx"] - 2, config["roi_y"]
            x_min, x_max = config["fit_window"]
            
            # 1D-Schnitt extrahieren
            p_epi = np.mean(sigma_epi[z, y_min:y_max, x_min:x_max], axis=0)
            
            if len(p_epi) == 100:
                raw_profiles[s_id][p_id] = p_epi

    aligned_profiles = defaultdict(list)

    print("Zentriere Defekte basierend auf strukturellem Konsens...")
    for s_id, p_dict in raw_profiles.items():
        all_model_profiles = list(p_dict.values())
        if len(all_model_profiles) == 0: continue
        
        # 1. Durchschnitt über alle Modelle bilden (Eliminiert Modell-Noise)
        mean_consensus_profile = np.mean(all_model_profiles, axis=0)
        
        # 2. Kurve leicht glätten (Eliminiert Pixel-Noise)
        smoothed_consensus = np.convolve(mean_consensus_profile, np.ones(5)/5, mode='same')
        
        # 3. Wahres, strukturelles Maximum suchen
        local_peak_idx = np.argmax(smoothed_consensus[30:70])
        global_peak_idx = local_peak_idx + 30
        
        # 4. Den exakten Shift berechnen
        shift_amount = 50 - global_peak_idx
        
        # 5. Alle Modelle für diese Serie identisch verschieben
        for p_id, prof in p_dict.items():
            aligned_profiles[p_id].append(apply_shift(prof, shift_amount))

    # =====================================================
    # 4. PLOTTING & STATISTIK (BEIDE ABSOLUTE UNCERTAINTY)
    # =====================================================
    results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    plt.subplots_adjust(top=0.85, wspace=0.25)
    
    x_axis = np.arange(100)
    colors = {"P02": "#1f77b4", "P14": "#ff7f0e", "P23": "#2ca02c", "CARE (MSE)": "#d62728"}

    for p_id, data_list in sorted(aligned_profiles.items()):
        if len(data_list) == 0: continue
            
        n_valid = np.sum(~np.isnan(np.stack(data_list)), axis=0)
        n_valid = np.where(n_valid == 0, 1, n_valid)
        
        # Ensemble Mean (Absolute Epistemic Uncertainty)
        avg_epi = np.nanmean(np.stack(data_list), axis=0)
        
        # Standard Error of the Mean (SEM)
        err_epi = np.nanstd(np.stack(data_list), axis=0) / np.sqrt(n_valid)
        
        c = colors.get(p_id, "black")
        
        # --- LEFT PLOT: Shaded Line Plot ---
        ax1.plot(x_axis, avg_epi, color=c, lw=2.5, label=p_id)
        ax1.fill_between(x_axis, avg_epi - err_epi, avg_epi + err_epi, color=c, alpha=0.15)
        
        # --- RIGHT PLOT: Scatter + Gauss Fit ---
        popt, perr = fit_ensemble_profile(x_axis, avg_epi, err_epi)
        
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

    # Styling Left Plot
    ax1.set_title("Absolute Epistemic Uncertainty\n(Shaded Standard Error)", fontsize=14, pad=15)
    ax1.set_ylabel("Standard Deviation $\sigma$", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=11)

    # Styling Right Plot
    ax2.set_title("Absolute Epistemic Uncertainty\n(Scatter Data + Gaussian Fit)", fontsize=14, pad=15)
    ax2.set_ylabel("Standard Deviation $\sigma$", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=11)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Pixel Index across ROI (Defect dynamically aligned at 50)", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(5, 95) 

    plt.suptitle("Quantitative Uncertainty Analysis: Structural Defect Alignment", fontsize=18, fontweight='bold', y=0.98)

    print("\n--- STATISTISCHE AUSWERTUNG (AUS ENSEMBLE GAUSS-FIT PARAMETERN) ---")
    print(pd.DataFrame(results).to_string(index=False))

    plt.savefig(OUT_DIR / "Final_Centered_Comparison_AbsoluteOnly.png", bbox_inches='tight')
    plt.show()