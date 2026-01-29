#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib

# =====================================================
# 1. Pfade & Konfiguration
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_ROOT = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits_K_Direction"

MODELS = {
    "Rang_1": "Rang_1_unet_25d_TripleLoss_a0.33_b0.17_bf64_D5_20260121-090819_loss0.0518_val0.0510.keras",
    "Rang_2": "Rang_2_unet_25d_TripleLoss_a0.17_b0.0_bf64_D5_20260121-012804_loss0.0259_val0.0296.keras",
    "Rang_3": "Rang_3_unet_25d_TripleLoss_a0.33_b0.33_bf64_D5_20260121-100752_loss0.0626_val0.0610.keras",
    "Rang_4": "Rang_4_unet_25d_TripleLoss_RESCUE_a0.17_b0.17_bf64_D5_20260123-223753_loss0.0450_val0.0428.keras",
    "Rang_5": "Rang_5_unet_25d_TripleLoss_RESCUE_a0.33_b0.0_bf64_D5_20260123-233434_loss0.0356_val0.0404.keras",
    "Rang_6": "Rang_6_unet_25d_TripleLoss_a0.17_b0.67_bf64_D5_20260121-051812_loss0.0981_val0.0815.keras",
    "Rang_7": "Rang_7_unet_25d_DeepScan_a0.25_b0.0833_bf64_D5_20260127-093131_loss0.0383_val0.0410.keras",
    "Rang_8": "Rang_8_unet_25d_DeepScan_a0.25_b0.0_bf64_D5_20260126-122819_loss0.0304_val0.0353.keras",
    "Rang_9": "Rang_9_unet_25d_TripleLoss_RESCUE_a0.17_b0.5_bf64_D5_20260123-214154_loss0.0832_val0.0685.keras",
    "Rang_10": "Rang_10_unet_25d_DeepScan_a0.25_b0.1667_bf64_D5_20260126-094704_loss0.0461_val0.0468.keras",
}

# HINZUGEFÜGT: ylim_raw (Zeile 2) und ylim_sbr (Zeile 3) für jede Serie
ROI_Y = (0, 192)
FIT_WINDOW = (90, 130)

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
IMAGE_WIDTH = 240

# =====================================================
# 2. Hilfsfunktionen
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def get_background_std(image, roi_y, bg_coords):
    box_right = image[roi_y[0]:roi_y[1], bg_coords[0]:bg_coords[1]]
    return np.std(box_right)

def calculate_sbr_k_profiles(image, cfg, bg_coords, force_noise_std=None):
    roi_x = cfg["roi_x"]
    signal_slice = image[ROI_Y[0]:ROI_Y[1], roi_x[0]:roi_x[1]]
    n_signal_cols = signal_slice.shape[1]
    profile_signal = np.sum(signal_slice, axis=1)
    
    background_slice = image[ROI_Y[0]:ROI_Y[1], bg_coords[0]:bg_coords[1]]
    n_background_cols = background_slice.shape[1]
    
    profile_background_raw = np.sum(background_slice, axis=1)
    scale = n_signal_cols / n_background_cols if n_background_cols > 0 else 0
    profile_background = profile_background_raw * scale

    profile_net = profile_signal - profile_background
    denom = np.where(profile_background == 0, 1e-9, profile_background)
    profile_sbr = profile_net / denom

    pixel_noise_std = force_noise_std if force_noise_std is not None else np.std(background_slice)
    
    err_signal_sum = pixel_noise_std * np.sqrt(n_signal_cols)
    err_bg_sum     = pixel_noise_std * np.sqrt(n_background_cols) * scale
    scalar_err_net = np.sqrt(err_signal_sum**2 + err_bg_sum**2)
    
    rel_err_net = scalar_err_net / np.abs(np.where(profile_net == 0, 1.0, profile_net))
    rel_err_bg  = err_bg_sum / np.abs(denom)
    profile_sbr_error = np.abs(profile_sbr) * np.sqrt(rel_err_net**2 + rel_err_bg**2)
    
    return np.arange(ROI_Y[0], ROI_Y[1]), profile_signal, profile_background, profile_sbr, profile_sbr_error

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 3: 
        return None, None, None
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
# 3. Prozess-Funktion
# =====================================================
def process_combination(rank_name, series_id, cfg):
    npz_file = f"Pred_{rank_name}_D5_S{series_id}_FullSeries.npz"
    path = IN_DIR / npz_file
    if not path.exists(): return

    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    bg_l = min(IMAGE_WIDTH, cfg["roi_x"][1] + cfg["bg_gap"])
    bg_r = min(IMAGE_WIDTH, bg_l + 20)
    bg_coords = (bg_l, bg_r)

    gt_noise = get_background_std(imgs[2], ROI_Y, bg_coords) 
    results = []
    
    for i, img in enumerate(imgs):
        noise = gt_noise if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_k_profiles(img, cfg, bg_coords, force_noise_std=noise)
        
        # ÄNDERUNG: Low Count (i=0) wird nie gefittet, i=1 und i=2 immer
        fit_y, par, perr = (None, None, None)
        if i > 0:
            fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WINDOW) 
            
        results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    # Visualisierung (3x3 Layout)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        # --- ZEILE 0: BILDER ---
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_low, p_high), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        roi_x = cfg["roi_x"]
        roi_w = roi_x[1] - roi_x[0]
        roi_h = ROI_Y[1] - ROI_Y[0]
        ax.add_patch(patches.Rectangle((roi_x[0], ROI_Y[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((roi_x[0], FIT_WINDOW[0]), roi_w, FIT_WINDOW[1]-FIT_WINDOW[0], lw=0, fc='green', alpha=0.2))
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y[0]), bg_r-bg_l, roi_h, lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        # --- ZEILE 1: RAW INTENSITÄTEN (Individuelles ylim) ---
        ax2 = axes[1, i]
        ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.15)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(cfg.get("ylim_raw", (2.5, 7.0))) 
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)

        # --- ZEILE 2: SRBR & GAUSS FIT (Individuelles ylim) ---
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, "
                 f"σ={p[2]:.2f}±{e[2]:.2f})")
            ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        
        ax3.set_xlabel("Pixel Y")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlim(FIT_WINDOW)
        ax3.set_ylim(cfg.get("ylim_sbr", (-0.2, 0.5)))

    plt.tight_layout()
    save_dir = OUT_ROOT / f"Serie_{series_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_{rank_name}_S{series_id}_Slice{idx}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. Main
# =====================================================
def main():
    matplotlib.use("Agg") 
    sorted_series = sorted(SERIES_CONFIG.keys())
    for rank in MODELS.keys():
        print(f"Processing {rank}...")
        for s_id in sorted_series:
            cfg = SERIES_CONFIG[s_id]
            process_combination(rank, s_id, cfg)
    print("Fertig!")

if __name__ == "__main__":
    main()