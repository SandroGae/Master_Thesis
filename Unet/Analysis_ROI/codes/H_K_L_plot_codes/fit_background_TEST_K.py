#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib

# =====================================================
# 1. GLOBALER SETUP-MODUS
# =====================================================

MODE = "TEST_BEST" 

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")

if MODE == "TEST_BEST":
    print(">>> Modus: TEST_BEST (K-Profil Vergleich...)")
    MODELS = {
        "Beast_P00_S42": "P00_a0.0000_b0.0000_seed42_best_model.keras",
        "Beast_P00_S43": "P00_a0.0000_b0.0000_seed43_best_model.keras",
        "Beast_P00_S44": "P00_a0.0000_b0.0000_seed44_best_model.keras",
        # ... deine anderen Referenzmodelle ...
    }
    # Pfad zu deinen neuen .npz Dateien
    IN_DIR  = ROOT_DIR / "Unet" / "Analysis_ROI" / "Predictions_Raw"
    OUT_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "H_K_L_Plots" / "TEST_ANALYSIS_K"
else:
    MODELS = MODELS_INF_SEED
    IN_DIR  = ROOT_DIR / "Unet/Analysis_ROI/Prediction.npz/Predictions_Raw_21_33" 
    OUT_DIR = ROOT_DIR / "Unet/Analysis_ROI/H_K_L_Plots/Analysis_K_Direction_INF_SEED"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. SERIEN-KONFIGURATION (L-Richtung Limits übernommen)
# =====================================================
ROI_Y = (0, 192)
FIT_WINDOW = (90, 130)

SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "bg_gap": -211, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "bg_gap": 52,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "bg_gap": 64,   "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    15: {"slice_idx": 19, "roi_x": (141, 162), "bg_gap": -162, "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 7.5), "ylim_sbr": (-0.2, 0.5)},
    16: {"slice_idx": 17, "roi_x": (121, 142), "bg_gap": 78,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "bg_gap": -213, "vis_p": (0.5, 99.5), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.2, 0.5)},
    22: {"slice_idx": 17, "roi_x": (173, 194), "bg_gap": -183, "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "bg_gap": -80,  "vis_p": (0.5, 99.0), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
    35: {"slice_idx": 24, "roi_x": (106, 127), "bg_gap": 45,   "vis_p": (0.5, 98.0), "ylim_raw": (3.5, 6.0), "ylim_sbr": (-0.2, 0.5)},
    50: {"slice_idx": 13, "roi_x": (97, 118),  "bg_gap": 38,   "vis_p": (0.5, 98.5), "ylim_raw": (3.5, 6.5), "ylim_sbr": (-0.2, 0.5)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]
IMAGE_WIDTH = 240

# =====================================================
# 3. HILFSFUNKTIONEN
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def calculate_sbr_k_profiles(image, cfg, bg_coords, force_noise_std=None):
    roi_x = cfg["roi_x"]
    signal_slice = image[ROI_Y[0]:ROI_Y[1], roi_x[0]:roi_x[1]]
    profile_signal = np.sum(signal_slice, axis=1)
    
    background_slice = image[ROI_Y[0]:ROI_Y[1], bg_coords[0]:bg_coords[1]]
    profile_background_raw = np.sum(background_slice, axis=1)
    scale = signal_slice.shape[1] / background_slice.shape[1]
    profile_background = profile_background_raw * scale

    denom = np.where(profile_background == 0, 1e-9, profile_background)
    profile_sbr = (profile_signal - profile_background) / denom

    px_noise = force_noise_std if force_noise_std is not None else np.std(background_slice)
    err_net = np.sqrt((px_noise * np.sqrt(signal_slice.shape[1]))**2 + (px_noise * np.sqrt(background_slice.shape[1]) * scale)**2)
    rel_net = err_net / np.abs(np.where(profile_signal-profile_background == 0, 1.0, profile_signal-profile_background))
    rel_bg = (px_noise * np.sqrt(background_slice.shape[1]) * scale) / np.abs(denom)
    profile_sbr_error = np.abs(profile_sbr) * np.sqrt(rel_net**2 + rel_bg**2)
    
    return np.arange(ROI_Y[0], ROI_Y[1]), profile_signal, profile_background, profile_sbr, profile_sbr_error

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 3: return None, None, None
    p0 = [np.max(y_f), x_f[np.argmax(y_f)], 5.0]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], 15.0])
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=y_err[mask], absolute_sigma=True, bounds=bounds, maxfev=10000)
        return gaussian(x, *popt), popt, np.sqrt(np.diag(pcov))
    except: return None, None, None

# =====================================================
# 4. PROZESS-FUNKTION
# =====================================================
def process_combination(model_id, s_id, cfg):
    # Dateinamens-Logik für TEST_BEST
    if MODE == "TEST_BEST":
        path = IN_DIR / f"Pred_{model_id}_S{s_id}.npz"
    else:
        path = IN_DIR / f"Pred_{model_id}_D5_S{s_id}_FullSeries.npz"

    if not path.exists(): return

    data = np.load(path)
    idx = cfg["slice_idx"]
    imgs = [data['lc'][idx], data['pred'][idx], data['gt'][idx]]
    
    bg_l = min(IMAGE_WIDTH - 21, max(0, cfg["roi_x"][1] + cfg["bg_gap"]))
    bg_r = bg_l + 20
    bg_coords = (bg_l, bg_r)

    # GT Noise für die Prediction (Index 1) nutzen
    gt_noise = np.std(imgs[2][ROI_Y[0]:ROI_Y[1], bg_coords[0]:bg_coords[1]])
    
    results = []
    y_ax = np.arange(ROI_Y[0], ROI_Y[1])
    for i, img in enumerate(imgs):
        noise = gt_noise if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_k_profiles(img, cfg, bg_coords, noise)
        fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WINDOW) if i > 0 else (None, None, None)
        results.append({'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        # ZEILE 0: BILDER (mit grünem Fenster)
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_low, p_high), cmap="gray_r")
        ax.set_title(f"{TITLES[i]} (S{s_id})", fontsize=14, fontweight='bold')
        rx = cfg["roi_x"]; rw = rx[1]-rx[0]; rh = ROI_Y[1]-ROI_Y[0]
        
        # --- GRÜNES FIT-FENSTER IM BILD ---
        ax.add_patch(patches.Rectangle((rx[0], FIT_WINDOW[0]), rw, FIT_WINDOW[1]-FIT_WINDOW[0], lw=0, fc='green', alpha=0.25))
        ax.add_patch(patches.Rectangle((rx[0], ROI_Y[0]), rw, rh, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((bg_l, ROI_Y[0]), bg_r-bg_l, rh, lw=1, ec='red', fc='red', alpha=0.15))
        ax.axis('off')

        # ZEILE 1: RAW INTENSITÄTEN (mit grünem axvspan)
        ax2 = axes[1, i]
        ax2.plot(y_ax, results[i]['sig'], color='blue', alpha=0.7, label='Signal')
        ax2.plot(y_ax, results[i]['bg'], color='red', alpha=0.7, label='Background')
        ax2.axvspan(FIT_WINDOW[0], FIT_WINDOW[1], color='green', alpha=0.1, label='Fit Range')
        ax2.set_ylim(cfg.get("ylim_raw")); ax2.grid(True, alpha=0.3)
        if i == 0: ax2.set_ylabel("Counts")

        # ZEILE 2: SBR & FIT
        ax3 = axes[2, i]
        ax3.errorbar(y_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', color='black', alpha=0.6)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = f"Amp={p[0]:.2f}, Peak={p[1]:.1f}, σ={p[2]:.2f}"
            ax3.plot(y_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
            ax3.legend(fontsize=8)
        ax3.set_xlim(FIT_WINDOW); ax3.set_ylim(cfg.get("ylim_sbr")); ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("Pixel Y")
        if i == 0: ax3.set_ylabel("SBR")

    plt.tight_layout()
    save_dir = OUT_DIR / f"Serie_{s_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_K_{model_id}_S{s_id}.png", bbox_inches='tight'); plt.close(fig)
    print(f" OK: Analysis_K_{model_id}_S{s_id}.png")

# =====================================================
# 5. MAIN
# =====================================================
def main():
    matplotlib.use("Agg")
    for chosen_name in MODELS.keys():
        print(f"Processing {chosen_name}...")
        for s_id, cfg in sorted(SERIES_CONFIG.items()):
            process_combination(chosen_name, s_id, cfg)

if __name__ == "__main__":
    main()