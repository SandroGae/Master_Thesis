#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib # Für Backend-Einstellung


# Pfade & Konfiguration
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

# Serien-spezifische Einstellungen mit individuellem Visualisierungs-Perzentil
ROI_Y = (0, 192)
FIT_WINDOW = (90, 130)

# Serien-spezifische Daten
SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "bg_gap": -211, "vis_p": (0.5, 99.0)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "bg_gap": 52,   "vis_p": (0.5, 98.0)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "bg_gap": 64,   "vis_p": (0.5, 99.0)},
    13: {"slice_idx": 1,  "roi_x": (65, 86),   "bg_gap": 66,  "vis_p": (0.5, 95.0)},
    15: {"slice_idx": 19, "roi_x": (141, 162),   "bg_gap": -162,  "vis_p": (0.5, 99.0)},
    16: {"slice_idx": 17, "roi_x": (121, 142),   "bg_gap": 78,  "vis_p": (0.5, 98.0)},
    21: {"slice_idx": 19, "roi_x": (192, 213),   "bg_gap": -213,  "vis_p": (0.5, 99.5)},
    22: {"slice_idx": 17, "roi_x": (173, 194),   "bg_gap": -183,  "vis_p": (0.5, 98.5)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "bg_gap": -80,   "vis_p": (0.5, 99.0)},
    50: {"slice_idx": 13, "roi_x": (97, 118),   "bg_gap": 38,  "vis_p": (0.5, 98.5)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]
IMAGE_WIDTH = 240


# Hilfsfunktionen
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    """ Visualisiert das Bild basierend auf individuellen Perzentilen """
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def get_background_std(image, roi_y, bg_coords):
    box_right = image[roi_y[0]:roi_y[1], bg_coords[0]:bg_coords[1]]
    return np.std(box_right)

def calculate_sbr_k_profiles(image, cfg, bg_coords, force_noise_std=None):
    roi_x = cfg["roi_x"]
    # HIER: Nutze die globale Variable roi_y statt cfg["roi_y"]
    # roi_y ist oben im Skript definiert
    
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



# 3. Prozess-Funktion
def process_combination(rank_name, series_id, cfg):
    npz_file = f"Pred_{rank_name}_D5_S{series_id}_FullSeries.npz"
    path = IN_DIR / npz_file
    if not path.exists(): return

    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    # Background-Bereich berechnen (Breite 20 wie im Referenz-Code)
    bg_l = min(IMAGE_WIDTH, cfg["roi_x"][1] + cfg["bg_gap"])
    bg_r = min(IMAGE_WIDTH, bg_l + 20)
    bg_coords = (bg_l, bg_r)

    # Rauschen von Ground Truth (Index 2) holen
    gt_noise = get_background_std(imgs[2], ROI_Y, bg_coords) 
    results = []
    
    # 1. Daten berechnen & Fitten
    for i, img in enumerate(imgs):
        # Prediction (Index 1) nutzt GT Rauschen
        noise = gt_noise if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_k_profiles(img, cfg, bg_coords, force_noise_std=noise)
        
        fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WINDOW) 
        results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    # 2. Visualisierung (3x3 Layout)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        # --- ZEILE 0: BILDER MIT BOXEN ---
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_low, p_high), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        
        roi_x = cfg["roi_x"]
        roi_w = roi_x[1] - roi_x[0]
        roi_h = ROI_Y[1] - ROI_Y[0]
        
        # Blaue ROI Box
        ax.add_patch(patches.Rectangle((roi_x[0], ROI_Y[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        
        # Grüner Fit-Bereich (Overlay)
        fit_h = FIT_WINDOW[1] - FIT_WINDOW[0]
        ax.add_patch(patches.Rectangle((roi_x[0], FIT_WINDOW[0]), roi_w, fit_h, lw=0, fc='green', alpha=0.2))
        
        # Rote Background Box
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y[0]), bg_r-bg_l, roi_h, lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        # --- ZEILE 1: RAW INTENSITÄTEN ---
        ax2 = axes[1, i]
        ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.15)
        ax2.grid(True, alpha=0.3)
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(2.5, 7) # Dein Standard-Limit

        # --- ZEILE 2: SRBR & GAUSS FIT ---
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            # Detailliertes Label wie im Referenzcode
            l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, "
                 f"σ={p[2]:.2f}±{e[2]:.2f}, "
                 f"Max={np.max(results[i]['fit']):.2f})")
            ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        
        ax3.set_xlabel("Pixel Y")
        if i == 0: ax3.set_ylabel("SRBR: (Signal - Background) / Background")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlim(FIT_WINDOW)
        ax3.set_ylim(-0.2, 0.5)

    # 3. Speichern
    plt.tight_layout()
    save_dir = OUT_ROOT / f"Serie_{series_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"Analysis_{rank_name}_S{series_id}_Slice{idx}.png"
    fig.savefig(save_dir / out_name, bbox_inches='tight')
    plt.close(fig)



# Main
def main():
    # Korrekte Einstellung für Headless-Server oder Batch-Processing
    matplotlib.use("Agg") 
    
    # Sortiert nach Serien-Nummer aufsteigend
    sorted_series = sorted(SERIES_CONFIG.keys())
    
    for rank in MODELS.keys():
        print(f"Processing {rank}...")
        for s_id in sorted_series:
            cfg = SERIES_CONFIG[s_id]
            process_combination(rank, s_id, cfg)
            
    print("Fertig! Alle Plots sind unter Analysis_ROI/Gaussian_fits_K_Direction gespeichert.")

if __name__ == "__main__":
    main()