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
FIT_WIN = (2, 38) # Fenster für die Z-Achse (0-40 Slices)
IMAGE_WIDTH = 240
BG_BOX_HEIGHT = 10

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    15: {"slice_idx": 19, "roi_x": (141, 162), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    16: {"slice_idx": 17, "roi_x": (121, 142), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    22: {"slice_idx": 17, "roi_x": (173, 194), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    35: {"slice_idx": 24, "roi_x": (106, 127), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    50: {"slice_idx": 13, "roi_x": (97, 118),  "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
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

def get_bg_coords_vertical(cfg):
    y_start, y_end = cfg["roi_y"]
    gap = cfg["bg_gap"]
    t_y2 = max(0, y_start - gap)
    t_y1 = max(0, t_y2 - BG_BOX_HEIGHT)
    b_y1 = min(192, y_end + gap)
    b_y2 = min(192, b_y1 + BG_BOX_HEIGHT)
    return (t_y1, t_y2), (b_y1, b_y2)

def calculate_sbr_z_profiles(volume, cfg, bg_coords, force_noise_std_array=None):
    n_frames = volume.shape[0]
    x1, x2 = cfg["roi_x"]; y1, y2 = cfg["roi_y"]
    (ty1, ty2), (by1, by2) = bg_coords
    n_sig_pixels = (x2 - x1) * (y2 - y1)
    
    p_sig, p_bg, p_sbr, p_err = [], [], [], []
    for i in range(n_frames):
        img = volume[i]
        sum_signal = np.sum(img[y1:y2, x1:x2])
        
        bg_list = []
        if ty2 > ty1: bg_list.append(img[ty1:ty2, x1:x2])
        if by2 > by1: bg_list.append(img[by1:by2, x1:x2])
        bg_concat = np.concatenate(bg_list)
        
        mean_bg = np.mean(bg_concat)
        scale = n_sig_pixels / bg_concat.size
        sum_bg_equiv = mean_bg * n_sig_pixels
        
        net_signal = sum_signal - sum_bg_equiv
        sbr = net_signal / max(sum_bg_equiv, 1e-9)
        
        # Noise-Statistik (GT-Noise falls vorhanden)
        px_std = force_noise_std_array[i] if force_noise_std_array is not None else np.std(bg_concat)
        
        # Gaußsche Fehlerfortpflanzung für Z-Profil
        err_sig = px_std * np.sqrt(n_sig_pixels)
        err_bg = px_std * np.sqrt(bg_concat.size) * scale
        err_net = np.sqrt(err_sig**2 + err_bg**2)
        
        # Relative Fehlerfortpflanzung Quotient
        total_rel = np.sqrt((err_net / max(abs(net_signal), 1e-9))**2 + (err_bg / max(sum_bg_equiv, 1e-9))**2)
        
        p_sig.append(sum_signal); p_bg.append(sum_bg_equiv)
        p_sbr.append(sbr); p_err.append(abs(sbr) * total_rel)
        
    return np.arange(n_frames), np.array(p_sig), np.array(p_bg), np.array(p_sbr), np.array(p_err)

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
    except: return None, None, None

# =====================================================
# 3. PROZESS-FUNKTION
# =====================================================
def process_combination(npz_path, s_id, cfg):
    data = np.load(npz_path)
    
    p_id_match = re.search(r"P(\d+)_", npz_path.stem)
    p_id = int(p_id_match.group(1)) if p_id_match else 0
    suffix_match = re.search(r"a\d+\.\d+_b\d+\.\d+_seed\d+", npz_path.stem)
    suffix = suffix_match.group(0) if suffix_match else "unknown"
    full_label = f"P{p_id:02d}_{suffix}"

    volumes = [data['lc'], data['pred'], data['gt']]
    bg_coords = get_bg_coords_vertical(cfg)
    
    # GT Noise Profil zur Stabilisierung der Prediction-Fehler
    gt_noise_z = []
    for v in volumes[2]:
        (ty1, ty2), (by1, by2) = bg_coords
        bg_list = []
        if ty2 > ty1: bg_list.append(v[ty1:ty2, cfg["roi_x"][0]:cfg["roi_x"][1]])
        if by2 > by1: bg_list.append(v[by1:by2, cfg["roi_x"][0]:cfg["roi_x"][1]])
        gt_noise_z.append(np.std(np.concatenate(bg_list)))

    results = []
    for i, vol in enumerate(volumes):
        # Nutze GT-Noise für das Prediction-Profil (Index 1)
        noise = gt_noise_z if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_z_profiles(vol, cfg, bg_coords, noise)
        fit_y, par, perr = (None, None, None)
        if i > 0: fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WIN)
        results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    fig.suptitle(f"Analysis H-Dir (Z-Axis): {full_label}", fontsize=16, fontweight='bold')
    
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(volumes[i][cfg["slice_idx"]], p_low, p_high), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        
        x1, x2, y1, y2 = cfg["roi_x"][0], cfg["roi_x"][1], cfg["roi_y"][0], cfg["roi_y"][1]
        rw, rh = x2 - x1, y2 - y1
        ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=0, fc='green', alpha=0.3))
        ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=2, ec='blue', fc='none'))
        (ty1, ty2), (by1, by2) = bg_coords
        if ty2 > ty1: ax.add_patch(patches.Rectangle((x1, ty1), rw, ty2-ty1, lw=1, ec='red', fc='red', alpha=0.2))
        if by2 > by1: ax.add_patch(patches.Rectangle((x1, by1), rw, by2-by1, lw=1, ec='red', fc='red', alpha=0.2))
        ax.axis('off')

        ax2 = axes[1, i]
        ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WIN[0], FIT_WIN[1], color='green', alpha=0.15)
        ax2.grid(True, alpha=0.3)
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(cfg.get("ylim_raw", (40, 65)))

        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Amp={p[0]:.2f}$\pm${e[0]:.2f}, Peak={p[1]:.1f}$\pm${e[1]:.1f}, $\sigma$={p[2]:.2f})")
            ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        
        ax3.set_xlabel("Image Index (Z-Axis)")
        if i == 0: ax3.set_ylabel("SRBR")
        ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=8)
        ax3.set_ylim(cfg.get("ylim_sbr", (-0.1, 0.55)))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    series_dir = OUT_DIR / f"series_{s_id}" ; series_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(series_dir / f"Plot_H_P{p_id:02d}_{suffix}.png", bbox_inches='tight')
    plt.close(fig)

def main():
    matplotlib.use("Agg")
    all_npzs = sorted(list(IN_DIR.glob("*.npz")))
    print(f"Gefunden: {len(all_npzs)} NPZ-Dateien. Starte H-Plotting...")
    for i, npz_path in enumerate(all_npzs):
        try:
            s_id = int(npz_path.stem.split('_S')[-1])
            if s_id in SERIES_CONFIG:
                process_combination(npz_path, s_id, SERIES_CONFIG[s_id])
                if i % 100 == 0: print(f"Fortschritt: {i}/{len(all_npzs)} H-Plots fertig...")
        except Exception as e:
            print(f"Fehler bei {npz_path.name}: {e}")

if __name__ == "__main__":
    main()