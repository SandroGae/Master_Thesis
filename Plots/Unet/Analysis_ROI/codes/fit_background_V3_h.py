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
OUT_ROOT = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits_Z_Direction"

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

# Config für Z-Profile (Z-Achse = Frame Index)
SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (190, 211), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 99.0), "fit_window": (2, 38)},
    11: {"slice_idx": 20, "roi_x": (83, 104),  "roi_y": (100, 119), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 98.0), "fit_window": (2, 38)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 99.0), "fit_window": (2, 38)},
    13: {"slice_idx": 1,  "roi_x": (65, 86),   "roi_y": (98, 113),  "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 95.0), "fit_window": (2, 38)},
    15: {"slice_idx": 19, "roi_x": (141, 162), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 99.0), "fit_window": (2, 38)},
    16: {"slice_idx": 17, "roi_x": (121, 142), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 98.0), "fit_window": (2, 38)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "roi_y": (101, 118), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 99.5), "fit_window": (2, 38)},
    22: {"slice_idx": 17, "roi_x": (173, 194), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 98.5), "fit_window": (2, 38)},
    29: {"slice_idx": 25, "roi_x": (59, 80),   "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 99.0), "fit_window": (2, 38)},
    50: {"slice_idx": 13, "roi_x": (97, 118),  "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "vis_p": (0.5, 98.5), "fit_window": (2, 38)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]

# =====================================================
# 2. Hilfsfunktionen
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def get_bg_coords(cfg):
    r1_bot = max(0, cfg["roi_y"][0] - cfg["bg_gap"])
    r1_top = max(0, r1_bot - cfg["bg_h"])
    r2_top = cfg["roi_y"][1] + cfg["bg_gap"]
    r2_bot = r2_top + cfg["bg_h"]
    return ((r1_top, r1_bot), (r2_top, r2_bot))

def get_gt_noise_std_series(volume, cfg, bg_coords):
    std_list = []
    (r1_t, r1_b), (r2_t, r2_b) = bg_coords
    for i in range(volume.shape[0]):
        img = volume[i]
        bg_px = np.concatenate([img[r1_t:r1_b, cfg["roi_x"][0]:cfg["roi_x"][1]], 
                                img[r2_t:r2_b, cfg["roi_x"][0]:cfg["roi_x"][1]]])
        std_list.append(np.std(bg_px))
    return np.array(std_list)

def calculate_sbr_z_profiles(volume, cfg, bg_coords, force_noise_std_array=None):
    n_frames = volume.shape[0]
    n_sig_px = (cfg["roi_x"][1]-cfg["roi_x"][0]) * (cfg["roi_y"][1]-cfg["roi_y"][0])
    (r1_t, r1_b), (r2_t, r2_b) = bg_coords
    
    prof_sig, prof_bg, prof_sbr, prof_err = [], [], [], []

    for i in range(n_frames):
        img = volume[i]
        sum_sig = np.sum(img[cfg["roi_y"][0]:cfg["roi_y"][1], cfg["roi_x"][0]:cfg["roi_x"][1]])
        
        bg_slice = np.concatenate([img[r1_t:r1_b, cfg["roi_x"][0]:cfg["roi_x"][1]], 
                                   img[r2_t:r2_b, cfg["roi_x"][0]:cfg["roi_x"][1]]])
        mean_bg = np.mean(bg_slice)
        sum_bg_equiv = mean_bg * n_sig_px
        
        net_sig = sum_sig - sum_bg_equiv
        denom = max(sum_bg_equiv, 1e-9)
        sbr_val = net_sig / denom
        
        # Fehler
        px_std = force_noise_std_array[i] if force_noise_std_array is not None else np.std(bg_slice)
        scale = n_sig_px / bg_slice.size
        err_sig = px_std * np.sqrt(n_sig_px)
        err_bg = px_std * np.sqrt(bg_slice.size) * scale
        err_net = np.sqrt(err_sig**2 + err_bg**2)
        
        total_rel = np.sqrt((err_net/abs(max(net_sig, 1e-9)))**2 + (err_bg/denom)**2)
        
        prof_sig.append(sum_sig); prof_bg.append(sum_bg_equiv)
        prof_sbr.append(sbr_val); prof_err.append(abs(sbr_val) * total_rel)

    return np.arange(n_frames), np.array(prof_sig), np.array(prof_bg), np.array(prof_sbr), np.array(prof_err)

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f = x[mask], y[mask]
    if len(y_f) < 5 or (np.max(y_f) - np.min(y_f)) < 0.05: return None, None, None
    
    p0 = [np.max(y_f)-np.median(y_f), x_f[np.argmax(y_f)], (fit_window[1]-fit_window[0])*0.15]
    bounds = ([0, fit_window[0], 0.5], [np.inf, fit_window[1], (fit_window[1]-fit_window[0])*0.4])
    
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=y_err[mask], absolute_sigma=True, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        res = y_f - gaussian(x_f, *popt)
        if perr[0]>0.5*popt[0] or perr[2]>0.5*popt[2] or popt[0]<3.0*np.std(res) or perr[1]>popt[2]: return None, None, None
        return gaussian(x, *popt), popt, perr
    except: return None, None, None

# =====================================================
# 3. Prozess-Funktion (VISUELL OPTIMIERT)
# =====================================================
def process_combination(rank_name, s_id, cfg):
    path = IN_DIR / f"Pred_{rank_name}_D5_S{s_id}_FullSeries.npz"
    if not path.exists(): return
    data = np.load(path)
    volumes = [data['lc'], data['pred'], data['gt']]
    
    bg_coords = get_bg_coords(cfg)
    gt_noise_series = get_gt_noise_std_series(volumes[2], cfg, bg_coords)
    
    results = []
    for i, vol in enumerate(volumes):
        noise = gt_noise_series if i == 1 else None
        x, sig, bg, sbr, err = calculate_sbr_z_profiles(vol, cfg, bg_coords, force_noise_std_array=noise)
        fit_y, par, perr = perform_gaussian_fit(x, sbr, err, cfg["fit_window"])
        results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    p_l, p_h = cfg.get("vis_p", (0.5, 99.5))

    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(volumes[i][cfg["slice_idx"]], p_l, p_h), cmap="gray_r")
        ax.set_title(f"{TITLES[i]} (S{s_id})", fontsize=14, fontweight='bold')
        
        roi_w, roi_h = cfg["roi_x"][1]-cfg["roi_x"][0], cfg["roi_y"][1]-cfg["roi_y"][0]
        ax.add_patch(patches.Rectangle((cfg["roi_x"][0], cfg["roi_y"][0]), roi_w, roi_h, lw=0, fc='green', alpha=0.3))
        ax.add_patch(patches.Rectangle((cfg["roi_x"][0], cfg["roi_y"][0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        (r1_t, r1_b), (r2_t, r2_b) = bg_coords
        ax.add_patch(patches.Rectangle((cfg["roi_x"][0], r1_t), roi_w, r1_b-r1_t, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((cfg["roi_x"][0], r2_t), roi_w, r2_b-r2_t, lw=1, ec='red', fc='red', alpha=0.2))
        ax.axis('off')

        ax2 = axes[1, i]
        ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(cfg["fit_window"][0], cfg["fit_window"][1], color='green', alpha=0.15)
        ax2.grid(True, alpha=0.3)
        if i==0: ax2.set_ylabel("Counts")
        if i==1: ax2.legend(loc='upper right', fontsize=8)
        # Dynamisches Scaling für Zeile 1
        y_min, y_max = np.min(results[i]['sig']), np.max(results[i]['sig'])
        ax2.set_ylim(y_min*0.9, y_max*1.1)

        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6)
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, σ={p[2]:.2f}±{e[2]:.2f}, Max={np.max(results[i]['fit']):.2f})")
            ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        ax3.set_xlabel("Image Index (Z-Axis)")
        if i==0: ax3.set_ylabel("SRBR")
        ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=7)
        ax3.set_xlim(0, results[i]['x'][-1]); ax3.set_ylim(-0.1, 0.6)

    plt.tight_layout()
    save_dir = OUT_ROOT / f"Serie_{s_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_Z_{rank_name}_S{s_id}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. Main Loop
# =====================================================
def main():
    matplotlib.use("Agg")
    for rank in MODELS.keys():
        print(f"Processing {rank} (Z-Direction)...")
        for s_id, cfg in SERIES_CONFIG.items():
            process_combination(rank, s_id, cfg)
    print("Fertig! Alle Z-Plots gespeichert.")

if __name__ == "__main__":
    main()