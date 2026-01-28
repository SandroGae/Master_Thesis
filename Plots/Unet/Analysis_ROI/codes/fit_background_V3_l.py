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
OUT_ROOT = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits_L_Direction"

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

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (140, 240)},
    11: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 119), "bg_gap": 5, "vis_p": (0.5, 98.0), "fit_window": (43, 143)},
    12: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (20, 120)},
    13: {"slice_idx": 1,  "roi_x": (0, 240), "roi_y": (98, 113),  "bg_gap": 5, "vis_p": (0.5, 95.0), "fit_window": (6, 106)},
    15: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (98, 198)},
    16: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "fit_window": (76, 176)},
    21: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (101, 118), "bg_gap": 5, "vis_p": (0.5, 99.5), "fit_window": (140, 240)},
    22: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "fit_window": (134, 234)},
    29: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (20, 120)},
    50: {"slice_idx": 13, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "fit_window": (52, 152)},
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

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit, y_fit = x[mask], y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    if len(y_fit) < 5 or (np.max(y_fit) - np.min(y_fit)) < 0.05:
        return None, None, None

    window_width = fit_window[1] - fit_window[0]
    p0 = [np.max(y_fit) - np.median(y_fit), x_fit[np.argmax(y_fit)], window_width * 0.15]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], window_width * 0.4])

    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, sigma=sigma_fit, absolute_sigma=True, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        residuals = y_fit - gaussian(x_fit, *popt)
        if perr[0] > 0.5*popt[0] or perr[2] > 0.5*popt[2] or popt[0] < 3.0*np.std(residuals) or perr[1] > popt[2]:
            return None, None, None
        return gaussian(x, *popt), popt, perr
    except:
        return None, None, None

# =====================================================
# 3. Prozess-Funktion (VISUELL OPTIMIERT)
# =====================================================
def process_combination(rank_name, s_id, cfg):
    path = IN_DIR / f"Pred_{rank_name}_D5_S{s_id}_FullSeries.npz"
    if not path.exists(): return
    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]

    # --- Geometrie & Background Logik ---
    roi_x = cfg["roi_x"]
    roi_y = cfg["roi_y"]
    roi_h = roi_y[1] - roi_y[0]
    bg_box_h = int(roi_h / 2) # Summe der 2 BG Boxen = ROI Höhe
    
    r1_bot = max(0, roi_y[0] - cfg["bg_gap"])
    r1_top = max(0, r1_bot - bg_box_h)
    r2_top = min(192, roi_y[1] + cfg["bg_gap"])
    r2_bot = min(192, r2_top + bg_box_h)
    bg_coords = ((r1_top, r1_bot), (r2_top, r2_bot))

    # GT Rauschen (Referenz für Prediction Plots)
    gt_bg1 = imgs[2][r1_top:r1_bot, roi_x[0]:roi_x[1]]
    gt_bg2 = imgs[2][r2_top:r2_bot, roi_x[0]:roi_x[1]]
    gt_noise = np.std(np.concatenate([gt_bg1, gt_bg2]))

    results = []
    x_axis = np.arange(roi_x[0], roi_x[1])

    for i, img in enumerate(imgs):
        noise = gt_noise if i == 1 else None
        # SBR Berechnung
        signal_slice = img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
        bg_slice = np.concatenate([img[r1_top:r1_bot, roi_x[0]:roi_x[1]], 
                                   img[r2_top:r2_bot, roi_x[0]:roi_x[1]]], axis=0)
        
        prof_sig = np.sum(signal_slice, axis=0)
        prof_bg_raw = np.sum(bg_slice, axis=0)
        scale = signal_slice.shape[0] / bg_slice.shape[0]
        prof_bg = prof_bg_raw * scale
        
        sbr = (prof_sig - prof_bg) / np.where(prof_bg == 0, 1e-9, prof_bg)
        
        # Fehlerberechnung
        px_std = noise if noise is not None else np.std(bg_slice)
        err_net = np.sqrt((px_std * np.sqrt(signal_slice.shape[0]))**2 + (px_std * np.sqrt(bg_slice.shape[0]) * scale)**2)
        rel_err_net = err_net / np.abs(np.where((prof_sig-prof_bg)==0, 1.0, (prof_sig-prof_bg)))
        rel_err_bg = (px_std * np.sqrt(bg_slice.shape[0]) * scale) / np.abs(np.where(prof_bg==0, 1e-9, prof_bg))
        sbr_err = np.abs(sbr) * np.sqrt(rel_err_net**2 + rel_err_bg**2)

        fit_y, par, perr = perform_gaussian_fit(x_axis, sbr, sbr_err, cfg["fit_window"])
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    # --- PLOTTING (Schönes Layout) ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_l, p_h = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        # 1. Bilder mit Boxen
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_l, p_h), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.add_patch(patches.Rectangle((roi_x[0], roi_y[0]), roi_x[1]-roi_x[0], roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((roi_x[0], r1_top), roi_x[1]-roi_x[0], bg_box_h, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((roi_x[0], r2_top), roi_x[1]-roi_x[0], bg_box_h, lw=1, ec='red', fc='red', alpha=0.2))
        fit_w = cfg["fit_window"][1] - cfg["fit_window"][0]
        ax.add_patch(patches.Rectangle((cfg["fit_window"][0], roi_y[0]), fit_w, roi_h, lw=0, fc='green', alpha=0.2))
        ax.axis('off')

        # 2. Raw Intensitäten
        ax2 = axes[1, i]
        ax2.plot(x_axis, results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(x_axis, results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(cfg["fit_window"][0], cfg["fit_window"][1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(1.5, 6)
        if i==0: ax2.set_ylabel("Counts")
        if i==1: ax2.legend(loc='upper right', fontsize=8)

        # 3. SRBR + Gaussian Fit
        ax3 = axes[2, i]
        ax3.errorbar(x_axis, results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, "
                 f"σ={p[2]:.2f}±{e[2]:.2f}, Max={np.max(results[i]['fit']):.2f})")
            ax3.plot(x_axis, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        ax3.set_xlim(cfg["fit_window"]); ax3.set_ylim(-0.1, 0.5)
        ax3.set_xlabel("Pixel X")
        ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    save_dir = OUT_ROOT / f"Serie_{s_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_L_{rank_name}_S{s_id}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. Main & Checksumme
# =====================================================
def main():
    matplotlib.use("Agg")
    for rank_name, model_file in MODELS.items():
        print(f"\nProcessing {rank_name}...")
        print(f"{'Serie':<6} | {'Low Count':<18} | {'Prediction':<18} | {'Ground Truth':<18}")
        print("-" * 70)
        
        for s_id in sorted(SERIES_CONFIG.keys()):
            cfg = SERIES_CONFIG[s_id]
            process_combination(rank_name, s_id, cfg)
            
            # Checksumme berechnen für Terminal-Output
            path = IN_DIR / f"Pred_{rank_name}_D5_S{s_id}_FullSeries.npz"
            if path.exists():
                d = np.load(path)
                si = cfg["slice_idx"]
                s_lc, s_pr, s_gt = np.sum(d['lc'][si]), np.sum(d['pred'][si]), np.sum(d['gt'][si])
                print(f"{s_id:<6} | {s_lc:<18.4f} | {s_pr:<18.4f} | {s_gt:<18.4f}")

    print("\nFertig! Alle L-Plots und Checksummen erstellt.")

if __name__ == "__main__":
    main()