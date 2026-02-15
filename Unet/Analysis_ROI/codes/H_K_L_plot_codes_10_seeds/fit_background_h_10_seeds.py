#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib
import os

# =====================================================
# 1. KONFIGURATION (Cluster-Pfade)
# =====================================================
MODELS_ROOT  = Path.home() / "scratch" / "43_Models_10_Seeds"
IN_DIR       = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"
OUT_DIR      = Path.home() / "scratch" / "Evaluation_Pipeline" / "Plots"

# Serien-Konfiguration (Original-Werte)
SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (195, 216), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    11: {"slice_idx": 20, "roi_x": (76, 97),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    15: {"slice_idx": 19, "roi_x": (136, 157), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    16: {"slice_idx": 17, "roi_x": (115, 136), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    22: {"slice_idx": 17, "roi_x": (176, 197), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    29: {"slice_idx": 25, "roi_x": (50, 71),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    35: {"slice_idx": 24, "roi_x": (128, 149), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    50: {"slice_idx": 13, "roi_x": (92, 113),  "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
}

FIX_W, FIX_H = 21, 11
BG_BOX_HEIGHT = 10
FIT_WIN       = (2, 38)
FIT_COLORS    = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES        = ["Low Count", "Prediction", "Ground Truth"]

# =====================================================
# 2. HILFSFUNKTIONEN (Haargenau wie im H-Skript)
# =====================================================
def update_config_to_fixed_size(cfg_dict, w, h):
    new_cfg = {}
    for s_id, vals in cfg_dict.items():
        v = vals.copy()
        cx = (v["roi_x"][0] + v["roi_x"][1]) / 2
        v["roi_x"] = (int(cx - w//2), int(cx - w//2 + w))
        cy = (v["roi_y"][0] + v["roi_y"][1]) / 2
        v["roi_y"] = (int(cy - h//2), int(cy - h//2 + h))
        new_cfg[s_id] = v
    return new_cfg

SERIES_CONFIG = update_config_to_fixed_size(SERIES_CONFIG, FIX_W, FIX_H)

def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low, p_high):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def get_bg_coords_vertical(cfg):
    y_start, y_end = cfg["roi_y"]
    gap = cfg["bg_gap"]
    t_y2 = max(0, y_start - gap); t_y1 = max(0, t_y2 - BG_BOX_HEIGHT)
    b_y1 = min(192, y_end + gap); b_y2 = min(192, b_y1 + BG_BOX_HEIGHT)
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
        bg_pixels = []
        if ty2 > ty1: bg_pixels.append(img[ty1:ty2, x1:x2])
        if by2 > by1: bg_pixels.append(img[by1:by2, x1:x2])
        bg_concat = np.concatenate(bg_pixels)
        mean_bg = np.mean(bg_concat)
        sum_bg_equiv = mean_bg * n_sig_pixels
        net_signal = sum_signal - sum_bg_equiv
        sbr = net_signal / max(sum_bg_equiv, 1e-9)
        px_std = force_noise_std_array[i] if force_noise_std_array is not None else np.std(bg_concat)
        scale = n_sig_pixels / bg_concat.size
        err_sig, err_bg = px_std * np.sqrt(n_sig_pixels), px_std * np.sqrt(bg_concat.size) * scale
        err_net = np.sqrt(err_sig**2 + err_bg**2)
        total_rel = np.sqrt((err_net/max(abs(net_signal),1e-9))**2 + (err_bg/max(sum_bg_equiv,1e-9))**2)
        p_sig.append(sum_signal); p_bg.append(sum_bg_equiv); p_sbr.append(sbr); p_err.append(abs(sbr) * total_rel)
    return np.arange(n_frames), np.array(p_sig), np.array(p_bg), np.array(p_sbr), np.array(p_err)

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 3: return None, None, None
    p0 = [np.max(y_f) - np.median(y_f), x_f[np.argmax(y_f)], 5.0]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], 15.0])
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=y_err[mask], absolute_sigma=True, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except: return None, None, None

# =====================================================
# 3. PROZESS-FUNKTION
# =====================================================
def process_combination(model_id, s_id, cfg):
    path = IN_DIR / f"Eval_{model_id}_S{s_id}.npz"
    if not path.exists(): return

    data = np.load(path)
    volumes = [data['lc'], data['pred'], data['gt']]
    bg_coords = get_bg_coords_vertical(cfg)
    
    # 3D Rausch-Profil vom GT extrahieren
    gt_noise = []
    for v_slice in volumes[2]:
        (ty1, ty2), (by1, by2) = bg_coords
        bg_list = []
        if ty2 > ty1: bg_list.append(v_slice[ty1:ty2, cfg["roi_x"][0]:cfg["roi_x"][1]])
        if by2 > by1: bg_list.append(v_slice[by1:by2, cfg["roi_x"][0]:cfg["roi_x"][1]])
        gt_noise.append(np.std(np.concatenate(bg_list)))

    results = []
    for i, vol in enumerate(volumes):
        noise = gt_noise if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_z_profiles(vol, cfg, bg_coords, noise)
        fit_y, par, perr = (None, None, None)
        if i > 0: fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WIN)
        results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg["vis_p"]

    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(volumes[i][cfg["slice_idx"]], p_low, p_high), cmap="gray_r")
        ax.set_title(f"{TITLES[i]} (S{s_id})", fontsize=14, fontweight='bold')
        x1, x2 = cfg["roi_x"]; y1, y2 = cfg["roi_y"]; rw, rh = x2 - x1, y2 - y1
        ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=0, fc='green', alpha=0.3))
        ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=2, ec='blue', fc='none'))
        (ty1, ty2), (by1, by2) = bg_coords
        if ty2 > ty1: ax.add_patch(patches.Rectangle((x1, ty1), rw, ty2-ty1, lw=1, ec='red', fc='red', alpha=0.2))
        if by2 > by1: ax.add_patch(patches.Rectangle((x1, by1), rw, by2-by1, lw=1, ec='red', fc='red', alpha=0.2))
        ax.axis('off')

        ax2 = axes[1, i]; ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7); ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7)
        ax2.axvspan(FIT_WIN[0], FIT_WIN[1], color='green', alpha=0.15); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(cfg.get("ylim_raw", (40, 65)))

        ax3 = axes[2, i]; ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', color='black', alpha=0.6)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = f"Gauss (Amp={p[0]:.2f}, Peak={p[1]:.1f}, $\sigma$={p[2]:.2f})"
            ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        ax3.set_xlim(FIT_WIN); ax3.set_ylim(cfg.get("ylim_sbr", (-0.1, 0.55))); ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    series_dir = OUT_DIR / f"series_{s_id}"
    series_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(series_dir / f"Analysis_H_{model_id}_S{s_id}.png", bbox_inches='tight')
    plt.close(fig)
    print(f" OK: H-Dir series_{s_id}/{model_id}")

# =====================================================
# 4. MAIN
# =====================================================
def main():
    matplotlib.use('Agg')
    all_models = sorted(list(MODELS_ROOT.glob("Point_*/*.h5")))
    model_ids = [m.stem for m in all_models]
    print(f"Starte H-Plotting für {len(model_ids)} Modelle...")
    for model_id in model_ids:
        for s_id, cfg in sorted(SERIES_CONFIG.items()):
            process_combination(model_id, s_id, cfg)

if __name__ == "__main__":
    main()