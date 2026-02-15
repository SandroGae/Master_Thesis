#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib
import os

# =====================================================
# 1. KONFIGURATION (Pfade & Serien)
# =====================================================
MODELS_ROOT = Path.home() / "scratch" / "43_Models_10_Seeds"
IN_DIR      = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"
OUT_DIR     = Path.home() / "scratch" / "Evaluation_Pipeline" / "Plots"

# K-Richtung Spezifika
ROI_Y = (0, 192)
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
# 2. HILFSFUNKTIONEN (Exakt wie in deinem K-Skript)
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 3: return None, None, None
    p0 = [np.max(y_f) - np.median(y_f), x_f[np.argmax(y_f)], 5.0]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], 15.0])
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=y_err[mask], 
                               absolute_sigma=True, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except:
        return None, None, None

# =====================================================
# 3. PROZESS-FUNKTION (K-Richtung Logik)
# =====================================================
def process_combination(model_id, s_id, cfg):
    path = IN_DIR / f"Eval_{model_id}_S{s_id}.npz"
    if not path.exists(): return

    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    bg_l = min(IMAGE_WIDTH, cfg["roi_x"][1] + cfg["bg_gap"])
    bg_r = min(IMAGE_WIDTH, bg_l + 20)
    bg_coords = (bg_l, bg_r)

    # Noise-Referenz vom Ground Truth Hintergrund
    gt_bg_slice = imgs[2][ROI_Y[0]:ROI_Y[1], bg_l:bg_r]
    gt_noise = np.std(gt_bg_slice)
    
    results = []
    y_ax = np.arange(ROI_Y[0], ROI_Y[1])

    for i, img in enumerate(imgs):
        sig_s = img[ROI_Y[0]:ROI_Y[1], cfg["roi_x"][0]:cfg["roi_x"][1]]
        bg_s  = img[ROI_Y[0]:ROI_Y[1], bg_l:bg_r]
        
        prof_sig = np.sum(sig_s, axis=1) # Summe über Spalten -> vertikales Profil
        scale = sig_s.shape[1] / bg_s.shape[1]
        prof_bg = np.sum(bg_s, axis=1) * scale
        
        denom = np.where(prof_bg == 0, 1e-9, prof_bg)
        sbr = (prof_sig - prof_bg) / denom
        
        p_std = gt_noise if i == 1 else np.std(bg_s)
        err_net = np.sqrt((p_std * np.sqrt(sig_s.shape[1]))**2 + (p_std * np.sqrt(bg_s.shape[1]) * scale)**2)
        sbr_err = np.abs(sbr) * np.sqrt((err_net/np.abs(np.where(prof_sig-prof_bg==0,1,prof_sig-prof_bg)))**2 + (p_std*np.sqrt(bg_s.shape[1])*scale/np.abs(denom))**2)

        fit_y, par, perr = (None, None, None)
        if i > 0:
            fit_y, par, perr = perform_gaussian_fit(y_ax, sbr, sbr_err, FIT_WINDOW)
        
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_low, p_high), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        roi_x = cfg["roi_x"]
        roi_w, roi_h = roi_x[1] - roi_x[0], ROI_Y[1] - ROI_Y[0]
        ax.add_patch(patches.Rectangle((roi_x[0], ROI_Y[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((roi_x[0], FIT_WINDOW[0]), roi_w, FIT_WINDOW[1]-FIT_WINDOW[0], lw=0, fc='green', alpha=0.2))
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y[0]), bg_r-bg_l, roi_h, lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        ax2 = axes[1, i]; ax2.plot(y_ax, results[i]['sig'], color='blue', alpha=0.7); ax2.plot(y_ax, results[i]['bg'], color='red', alpha=0.7)
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.15)
        ax2.set_ylim(cfg.get("ylim_raw", (2.5, 7.0))); ax2.grid(True, alpha=0.3)

        ax3 = axes[2, i]; ax3.errorbar(y_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', color='black', alpha=0.6)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = f"Gauss (Amp={p[0]:.2f}, Peak={p[1]:.1f}, $\sigma$={p[2]:.2f})"
            ax3.plot(y_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        ax3.set_xlim(FIT_WINDOW); ax3.set_ylim(cfg.get("ylim_sbr", (-0.2, 0.5))); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    series_dir = OUT_DIR / f"series_{s_id}"
    series_dir.mkdir(parents=True, exist_ok=True)
    
    save_p = series_dir / f"Analysis_K_{model_id}_S{s_id}.png"
    fig.savefig(save_p, bbox_inches='tight')
    plt.close(fig)
    print(f" OK: K-Dir series_{s_id}/{save_p.name}")

# =====================================================
# 4. MAIN
# =====================================================
def main():
    matplotlib.use("Agg")
    all_models = sorted(list(MODELS_ROOT.glob("Point_*/*.h5")))
    model_ids = [m.stem for m in all_models]
    
    print(f"Starte K-Plotting für {len(model_ids)} Modelle...")

    for model_id in model_ids:
        for s_id, cfg in sorted(SERIES_CONFIG.items()):
            process_combination(model_id, s_id, cfg)

if __name__ == "__main__":
    main()