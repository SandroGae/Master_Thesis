#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib
import os
import re

# =====================================================
# 1. KONFIGURATION
# =====================================================
IN_DIR  = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"
OUT_DIR = Path.home() / "scratch" / "Evaluation_Pipeline" / "Plots"

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 99.0)},
    11: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 119), "bg_gap": 5, "bg_h": 10, "fit_window": (43, 143),  "y_lim_raw": (3.0, 8.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    12: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (24, 124),  "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 99.0)},
    15: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (98, 198),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 99.0)},
    16: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (76, 176),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    21: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (101, 118), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (3.0, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 99.5)},
    22: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (134, 234), "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.5)},
    29: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (20, 120),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 99.0)},
    35: {"slice_idx": 24, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (64, 164),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    50: {"slice_idx": 13, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 10, "bg_h": 10, "fit_window": (52, 152), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.5)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f, s_f = x[mask], y[mask], y_err[mask]
    if len(y_f) < 5: return None, None, None
    win_w = fit_window[1] - fit_window[0]
    p0 = [np.max(y_f) - np.median(y_f), x_f[np.argmax(y_f)], win_w * 0.15]
    bounds = ((0, fit_window[0], 0.5), (np.inf, fit_window[1], win_w * 0.4))
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=s_f, absolute_sigma=True, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except: 
        return None, None, None

# =====================================================
# 3. PROZESS-FUNKTION
# =====================================================
def process_combination(npz_path, s_id, cfg):
    data = np.load(npz_path)
    p_id = int(re.search(r"P(\d+)_", npz_path.stem).group(1))
    suffix = re.search(r"a\d+\.\d+.*", npz_path.stem).group(0)
    full_label = f"P{p_id:02d}_{suffix}"  # Rekonstruiert den sauberen Namen für Titel & Pfad
    
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]

    rx, ry, bg_h = cfg["roi_x"], cfg["roi_y"], cfg["bg_h"]
    r1_t = max(0, ry[0] - cfg["bg_gap"] - bg_h); r1_b = r1_t + bg_h
    r2_t = min(192, ry[1] + cfg["bg_gap"]); r2_b = min(192, r2_t + bg_h)

    gt_bg = np.concatenate([imgs[2][r1_t:r1_b, rx[0]:rx[1]], imgs[2][r2_t:r2_b, rx[0]:rx[1]]])
    gt_std = np.std(gt_bg)

    results = []
    x_ax = np.arange(rx[0], rx[1])

    for i, img in enumerate(imgs):
        sig_s = img[ry[0]:ry[1], rx[0]:rx[1]]
        bg_s = np.concatenate([img[r1_t:r1_b, rx[0]:rx[1]], img[r2_t:r2_b, rx[0]:rx[1]]])
        prof_sig = np.sum(sig_s, axis=0)
        scale = sig_s.shape[0] / bg_s.shape[0]
        prof_bg = np.sum(bg_s, axis=0) * scale
        denom = np.where(prof_bg == 0, 1e-9, prof_bg)
        sbr = (prof_sig - prof_bg) / denom
        
        p_std = gt_std if i == 1 else np.std(bg_s)
        err_net = np.sqrt((p_std * np.sqrt(sig_s.shape[0]))**2 + (p_std * np.sqrt(bg_s.shape[0]) * scale)**2)
        sbr_err = np.abs(sbr) * np.sqrt((err_net/np.abs(np.where(prof_sig-prof_bg==0,1,prof_sig-prof_bg)))**2 + (p_std*np.sqrt(bg_s.shape[0])*scale/np.abs(denom))**2)

        fit_y, par, perr = (None, None, None) if i == 0 else perform_gaussian_fit(x_ax, sbr, sbr_err, cfg["fit_window"])
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    fig.suptitle(f"Analysis L-Dir: {full_label}", fontsize=16, fontweight='bold')

    p_l, p_h = cfg.get("vis_p", (0.5, 99.5))
    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_l, p_h), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14)
        roi_w, roi_h = rx[1]-rx[0], ry[1]-ry[0]
        ax.add_patch(patches.Rectangle((rx[0], ry[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((rx[0], r1_t), roi_w, bg_h, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((rx[0], r2_t), roi_w, bg_h, lw=1, ec='red', fc='red', alpha=0.2))
        ax.axis('off')

        ax2 = axes[1, i]; ax2.plot(x_ax, results[i]['sig'], color='blue', alpha=0.7); ax2.plot(x_ax, results[i]['bg'], color='red', alpha=0.7)
        ax2.axvspan(cfg["fit_window"][0], cfg["fit_window"][1], color='green', alpha=0.1)
        ax2.set_ylim(cfg["y_lim_raw"]); ax2.grid(True, alpha=0.3)

        ax3 = axes[2, i]; ax3.errorbar(x_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', color='black', alpha=0.6)
        if results[i]['fit'] is not None:
            ax3.plot(x_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5)
        ax3.set_xlim(cfg["fit_window"]); ax3.set_ylim(cfg["y_lim_sbr"]); ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    series_dir = OUT_DIR / f"series_{s_id}"
    series_dir.mkdir(parents=True, exist_ok=True)
    
    save_p = series_dir / f"Plot_L_P{p_id:02d}_{suffix}.png"
    fig.savefig(save_p, bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. MAIN
# =====================================================
def main():
    matplotlib.use("Agg")
    all_npzs = sorted(list(IN_DIR.glob("*.npz")))
    
    print(f"Gefunden: {len(all_npzs)} NPZ-Dateien. Starte Plotting...")

    for i, npz_path in enumerate(all_npzs):
        try:
            # Extrahiert S_id aus dem Ende des Namens, z.B. ..._S50.npz
            s_str = npz_path.stem.split('_S')[-1]
            s_id = int(s_str)
            
            if s_id in SERIES_CONFIG:
                process_combination(npz_path, s_id, SERIES_CONFIG[s_id])
                if i % 50 == 0: print(f"Fortschritt: {i}/{len(all_npzs)} Plots fertig...")
        except Exception as e:
            print(f"Fehler bei {npz_path.name}: {e}")

if __name__ == "__main__":
    main()