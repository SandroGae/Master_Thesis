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
IN_DIR = Path.home() / "scratch" / "Evaluation_Pipeline" / "npz_files"
OUT_DIR = Path.home() / "scratch" / "Evaluation_Pipeline" / "Plots"

ROI_Y_LIMITS = (0, 192)
FIT_WINDOW = (90, 130)
IMAGE_WIDTH = 240

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "bg_gap": -211, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "bg_gap": 52,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "bg_gap": 64,   "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
    15: {"slice_idx": 19, "roi_x": (141, 162), "bg_gap": -162, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    16: {"slice_idx": 17, "roi_x": (121, 142), "bg_gap": 78,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "bg_gap": -213, "vis_p": (0.5, 99.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    22: {"slice_idx": 17, "roi_x": (173, 194), "bg_gap": -183, "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "bg_gap": -80,  "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    35: {"slice_idx": 24, "roi_x": (106, 127), "bg_gap": 45,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
    50: {"slice_idx": 13, "roi_x": (97, 118),  "bg_gap": 38,   "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
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
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=s_f, absolute_sigma=True, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except:
        return None, None, None

# =====================================================
# 3. PROZESS-FUNKTION
# =====================================================
def process_combination(npz_path, s_id, cfg):
    data = np.load(npz_path)
    
    # Sicherer Regex-Check
    p_id_match = re.search(r"P(\d+)_", npz_path.stem)
    p_id = int(p_id_match.group(1)) if p_id_match else 0
    suffix_match = re.search(r"a\d+\.\d+_b\d+\.\d+_seed\d+", npz_path.stem)
    suffix = suffix_match.group(0) if suffix_match else "unknown"
    full_label = f"P{p_id:02d}_{suffix}"
    
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    # Hintergrund-Bereich (K-Richtung: seitlicher Streifen)
    bg_l = min(IMAGE_WIDTH - 21, max(0, cfg["roi_x"][1] + cfg["bg_gap"]))
    bg_r = bg_l + 20
    
    gt_bg_area = imgs[2][ROI_Y_LIMITS[0]:ROI_Y_LIMITS[1], bg_l:bg_r]
    gt_std = np.std(gt_bg_area)
    
    results = []
    y_ax = np.arange(ROI_Y_LIMITS[0], ROI_Y_LIMITS[1])

    for i, img in enumerate(imgs):  
        # K-Dir: Vertikaler Schnitt (Profil über Y)
        sig_s = img[ROI_Y_LIMITS[0]:ROI_Y_LIMITS[1], cfg["roi_x"][0]:cfg["roi_x"][1]]
        bg_s  = img[ROI_Y_LIMITS[0]:ROI_Y_LIMITS[1], bg_l:bg_r]

        prof_sig = np.sum(sig_s, axis=1) # Horizontal summieren für Y-Profil
        scale = sig_s.shape[1] / bg_s.shape[1]
        prof_bg = np.sum(bg_s, axis=1) * scale
        
        denom = np.where(prof_bg == 0, 1e-9, prof_bg)
        sbr = (prof_sig - prof_bg) / denom
        
        p_std = gt_std if i == 1 else np.std(bg_s)
        
        # Gaußsche Fehlerfortpflanzung (Y-Richtung)
        n_sig_px = sig_s.shape[0] * sig_s.shape[1]
        n_bg_px = bg_s.shape[0] * bg_s.shape[1]
        
        # Fehler des Netto-Signals (Zähler)
        err_net = np.sqrt((p_std * np.sqrt(n_sig_px/sig_s.shape[0]))**2 + (p_std * np.sqrt(n_bg_px/bg_s.shape[0]) * scale)**2)
        diff = np.where(prof_sig - prof_bg == 0, 1, prof_sig - prof_bg)
        
        # Gesamtfehler SBR
        sbr_err = np.abs(sbr) * np.sqrt((err_net/np.abs(diff))**2 + (p_std * np.sqrt(n_bg_px/bg_s.shape[0]) * scale / np.abs(denom))**2)

        fit_y, par, perr = (None, None, None) if i == 0 else perform_gaussian_fit(y_ax, sbr, sbr_err, FIT_WINDOW)
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    fig.suptitle(f"Analysis K-Dir (Vertical): {full_label}", fontsize=16, fontweight='bold')
    
    p_l, p_h = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_l, p_h), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        rx = cfg["roi_x"]
        
        # ROI & BG Patches
        ax.add_patch(patches.Rectangle((rx[0], ROI_Y_LIMITS[0]), rx[1]-rx[0], ROI_Y_LIMITS[1]-ROI_Y_LIMITS[0], lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((rx[0], FIT_WINDOW[0]), rx[1]-rx[0], FIT_WINDOW[1]-FIT_WINDOW[0], lw=0, fc='green', alpha=0.2))
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y_LIMITS[0]), bg_r-bg_l, ROI_Y_LIMITS[1]-ROI_Y_LIMITS[0], lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        ax2 = axes[1, i]
        ax2.plot(y_ax, results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(y_ax, results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.15)
        ax2.set_ylim(cfg.get("ylim_raw", (3.5, 6.5))) 
        ax2.grid(True, alpha=0.3)
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)

        ax3 = axes[2, i]
        ax3.errorbar(y_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Amp={p[0]:.2f}±{e[0]:.2f}, Peak={p[1]:.1f}±{e[1]:.1f}, σ={p[2]:.2f})")
            ax3.plot(y_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        
        ax3.set_xlabel("Pixel Y")
        ax3.set_xlim(FIT_WINDOW)
        ax3.set_ylim(cfg.get("ylim_sbr", (-0.2, 0.5)))
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    series_dir = OUT_DIR / f"series_{s_id}"
    series_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(series_dir / f"Plot_K_P{p_id:02d}_{suffix}.png", bbox_inches='tight')
    plt.close(fig)

def main():
    matplotlib.use("Agg")
    all_npzs = sorted(list(IN_DIR.glob("*.npz")))
    print(f"Gefunden: {len(all_npzs)} NPZ-Dateien. Starte K-Plotting...")
    for i, npz_path in enumerate(all_npzs):
        try:
            s_id = int(npz_path.stem.split('_S')[-1])
            if s_id in SERIES_CONFIG:
                process_combination(npz_path, s_id, SERIES_CONFIG[s_id])
                if i % 100 == 0: print(f"Fortschritt: {i}/{len(all_npzs)} K-Plots fertig...")
        except Exception as e:
            print(f"Fehler bei {npz_path.name}: {e}")

if __name__ == "__main__":
    main()