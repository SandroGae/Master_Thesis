#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib

# =====================================================
# 1. GLOBALER TOGGLE (Hier auswählen!)
# =====================================================
USE_RERUN = True  # True für die 56 neuen Modelle, False für die ursprünglichen 10

MODELS_ORIGINAL = {
    "Rang_1": "Rang_1_unet_25d_TripleLoss_a0.33_b0.17_bf64_D5_20260121-090819_loss0.0518_val0.0510.keras",
    "Rang_2": "Rang_2_unet_25d_TripleLoss_a0.17_b0.0_bf64_D5_20260121-012804_loss0.0259_val0.0296.keras",
    "Rang_3": "Rang_3_unet_25d_TripleLoss_a0.33_b0.33_bf64_D5_20260121-100752_loss0.0626_val0.0610.keras",
    "Rang_4": "Rang_4_unet_25d_TripleLoss_RESCUE_a0.17_b0.17_bf64_D5_20260123-223753_loss0.0450_val0.0428.keras",
    "Rang_5": "Rang_5_unet_25d_TripleLoss_RESCUE_a0.33_b0.0_bf64_D5_20260123-233434_loss0.0356_val0.0404.keras",
    "Rang_6": "Rang_6_unet_25d_TripleLoss_a0.17_b0.67_bf64_D5_20260121-051812_loss0.0981_val0.0815.keras",
    "Rang_7": "Rang_7_unet_25d_DeepScan_a0.25_b0.0833_bf64_D5_20260127-093131_loss0.0383_val0.0410.keras",
    "Rang_8": "Rang_8_unet_25d_DeepScan_a0.25_b0.0_bf64_D5_20260126-122819_loss0.0304_val0.0353.keras",
    "Rang_9": "Rang_9_unet_25d_TripleLoss_RESCUE_a0.17_b0.5_bf64_D5_20260123-214154_loss0.0832_val0.0685.keras",
    "Rank_10": "Rang_10_unet_25d_DeepScan_a0.25_b0.1667_bf64_D5_20260126-094704_loss0.0461_val0.0468.keras",
}

MODELS_RERUN = {
    "Rank_01": "Rank_01__DeepScan_a0.1667_b0.0_seed42_20260128-234241_loss0.0257_val0.0294.keras",
    "Rank_02": "Rank_02__DeepScan_a0.25_b0.0_seed42_20260129-121519_loss0.0309_val0.0351.keras",
    "Rank_03": "Rank_03__DeepScan_a0.1667_b0.1667_seed42_20260129-010522_loss0.0407_val0.0425.keras",
    "Rank_04": "Rank_04__DeepScan_a0.25_b0.5_seed42_20260129-201142_loss0.0899_val0.0702.keras",
    "Rank_05": "Rank_05__DeepScan_a0.25_b0.0833_seed42_20260129-133505_loss0.0373_val0.0410.keras",
    "Rank_06": "Rank_06__DeepScan_a0.25_b0.3333_seed42_20260129-175254_loss0.0702_val0.0585.keras",
    "Rank_07": "Rank_07__DeepScan_a0.25_b0.25_seed42_20260129-162232_loss0.0504_val0.0527.keras",
    "Rank_08": "Rank_08__DeepScan_a0.25_b0.1667_seed42_20260129-150606_loss0.0476_val0.0469.keras",
    "Rank_09": "Rank_09__DeepScan_a0.3333_b0.1667_seed42_20260129-102303_loss0.0505_val0.0512.keras",
    "Rank_10": "Rank_10__DeepScan_a0.1667_b0.3333_seed42_20260129-023131_loss0.0558_val0.0555.keras",
    "Rank_11": "Rank_11__DeepScan_a0.3333_b0.3333_seed42_20260129-113724_loss0.0708_val0.0616.keras",
    "Rank_12": "Rank_12__DeepScan_a0.5_b0.0_seed42_20260129-160827_loss0.0461_val0.0520.keras",
    "Rank_13": "Rank_13__DeepScan_a0.3333_b0.0_seed42_20260129-090833_loss0.0361_val0.0409.keras",
    "Rank_14": "Rank_14__DeepScan_a0.1667_b0.5_seed42_20260129-035815_loss0.0709_val0.0685.keras",
    "Rank_15": "Rank_15__DeepScan_a0.25_b0.4167_seed42_20260129-190608_loss0.0762_val0.0645.keras",
    "Rank_16": "Rank_16__DeepScan_a0.25_b0.5833_seed42_20260129-212442_loss0.0832_val0.0761.keras",
    "Rank_17": "Rank_17__DeepScan_a0.3333_b0.5_seed42_20260129-124730_loss0.0838_val0.0720.keras",
    "Rank_18": "Rank_18__DeepScan_a0.1667_b0.6667_seed42_20260129-052209_loss0.1121_val0.0815.keras",
    "Rank_19": "Rank_19__DeepScan_a0.25_b0.6667_seed42_20260129-224041_loss0.1027_val0.0819.keras",
    "Rank_20": "Rank_20__DeepScan_a0.5_b0.1667_seed43_20260129-173735_loss0.0590_val0.0602.keras",
    "Rank_21": "Rank_21__DeepScan_a0.3333_b0.6667_seed42_20260129-140531_loss0.1004_val0.0824.keras",
    "Rank_22": "Rank_22__DeepScan_a0.6667_b0.0_seed43_20260130-005859_loss0.0558_val0.0636.keras",
    "Rank_23": "Rank_23__DeepScan_a0.6667_b0.1667_seed43_20260130-020940_loss0.0630_val0.0688.keras",
    "Rank_24": "Rank_24__DeepScan_a0.5_b0.3333_seed43_20260129-190228_loss0.0743_val0.0680.keras",
    "Rank_25": "Rank_25__DeepScan_a0.3333_b0.8333_seed45_20260129-205112_loss0.1053_val0.0927.keras",
    "Rank_26": "Rank_26__DeepScan_a0.5_b0.5_seed43_20260129-203141_loss0.0805_val0.0756.keras",
    "Rank_27": "Rank_27__DeepScan_a0.25_b0.75_seed42_20260129-234822_loss0.1124_val0.0878.keras",
    "Rank_28": "Rank_28__DeepScan_a0.5_b0.6667_seed43_20260129-215115_loss0.0971_val0.0836.keras",
    "Rank_29": "Rank_29__DeepScan_a0.1667_b0.8333_seed42_20260129-064731_loss0.1090_val0.0945.keras",
    "Rank_30": "Rank_30__DeepScan_a0.25_b0.8333_seed43_20260130-011305_loss0.1068_val0.0937.keras",
    "Rank_31": "Rank_31__DeepScan_a0.5_b0.8333_seed44_20260130-093833_loss0.0939_val0.0910.keras",
    "Rank_32": "Rank_32__DeepScan_a0.6667_b0.3333_seed43_20260130-032200_loss0.0723_val0.0740.keras",
    "Rank_33": "Rank_33__DeepScan_a0.25_b0.9167_seed43_20260130-024019_loss0.1209_val0.0995.keras",
    "Rank_34": "Rank_34__DeepScan_a0.8333_b0.0_seed43_20260129-192305_loss0.0659_val0.0751.keras",
    "Rank_35": "Rank_35__DeepScan_a0.6667_b0.6667_seed43_20260130-052955_loss0.0866_val0.0844.keras",
    "Rank_36": "Rank_36__DeepScan_a0.6667_b0.5_seed43_20260130-042141_loss0.0804_val0.0794.keras",
    "Rank_37": "Rank_37__DeepScan_a0.8333_b0.6667_seed44_20260129-224337_loss0.0808_val0.0849.keras",
    "Rank_38": "Rank_38__DeepScan_a0.8333_b0.3333_seed43_20260129-171101_loss0.0743_val0.0802.keras",
    "Rank_39": "Rank_39__DeepScan_a0.6667_b0.8333_seed43_20260129-210109_loss0.0942_val0.0897.keras",
    "Rank_40": "Rank_40__DeepScan_a0.8333_b0.5_seed43_20260129-160104_loss0.0764_val0.0828.keras",
    "Rank_41": "Rank_41__DeepScan_a0.3333_b1.0_seed44_20260129-214956_loss0.1431_val0.1031.keras",
    "Rank_42": "Rank_42__DeepScan_a0.1667_b1.0_seed42_20260129-075907_loss0.1454_val0.1075.keras",
    "Rank_43": "Rank_43__DeepScan_a0.25_b1.0_seed43_20260130-040610_loss0.1373_val0.1054.keras",
    "Rank_44": "Rank_44__DeepScan_a0.8333_b0.1667_seed43_20260129-182440_loss0.0702_val0.0777.keras",
    "Rank_45": "Rank_45__DeepScan_a0.8333_b0.8333_seed43_20260129-144838_loss0.0823_val0.0880.keras",
    "Rank_46": "Rank_46__DeepScan_a0.6667_b1.0_seed44_20260130-102728_loss0.0957_val0.0946.keras",
    "Rank_47": "Rank_47__DeepScan_a0.5_b1.0_seed43_20260129-233812_loss0.1115_val0.0991.keras",
    "Rank_48": "Rank_48__DeepScan_a0.8333_b1.0_seed43_20260129-133532_loss0.0888_val0.0906.keras",
    "Rank_49": "Rank_49__DeepScan_a1.0_b0.0_seed42_20260129-121932_loss0.0759_val0.0859.keras",
    "Rank_50": "Rank_50__DeepScan_a0.0_b0.0_seed42_20260128-145959_loss0.0199_val0.0225.keras",
    "Rank_51": "Rank_51__DeepScan_a0.0_b0.1667_seed42_20260128-161918_loss0.0421_val0.0374.keras",
    "Rank_52": "Rank_52__DeepScan_a0.0_b0.3333_seed42_20260128-174208_loss0.0645_val0.0523.keras",
    "Rank_53": "Rank_53__DeepScan_a0.0_b0.5_seed42_20260128-190417_loss0.0847_val0.0672.keras",
    "Rank_54": "Rank_54__DeepScan_a0.0_b0.6667_seed42_20260128-201608_loss0.1064_val0.0821.keras",
    "Rank_55": "Rank_55__DeepScan_a0.0_b0.8333_seed42_20260128-212613_loss0.1281_val0.0970.keras",
    "Rank_56": "Rank_56__DeepScan_a0.0_b1.0_seed42_20260128-223731_loss fiber0.1660_val0.1118.keras",
}

# AUTOMATISCHE PFAD-Zuweisung
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"

if USE_RERUN:
    print(">>> Nutze RERUN Setup (K-Richtung)")
    MODELS = MODELS_RERUN
    IN_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw_RERUN"
    # KORREKTUR: Eigener Ordner für die Plots
    OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Analysis_L_Direction_RERUN"
else:
    print(">>> Nutze ORIGINAL Setup (K-Richtung)")
    MODELS = MODELS_ORIGINAL
    IN_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw"
    OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Analysis_L_Direction"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hier kannst du nun für jedes Bild (Serie) die Y-Limits der unteren beiden Zeilen anpassen
SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5)},
    11: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 119), "bg_gap": 5, "vis_p": (0.5, 98.0), "fit_window": (43, 143),  "y_lim_raw": (3.0, 8.0), "y_lim_sbr": (-0.1, 0.5)},
    12: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (24, 124),  "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5)},
    15: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "fit_window": (98, 198),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5)},
    16: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "fit_window": (76, 176),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5)},
    21: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (101, 118), "bg_gap": 5, "vis_p": (0.5, 99.5), "fit_window": (140, 240), "y_lim_raw": (3.0, 5.5), "y_lim_sbr": (-0.1, 0.5)},
    22: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "fit_window": (134, 234), "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5)},
    29: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 3, "vis_p": (0.5, 99.0), "fit_window": (20, 120),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5)},
    35: {"slice_idx": 24, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "fit_window": (64, 164),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5)},
    50: {"slice_idx": 13, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 10, "vis_p": (0.5, 98.5), "fit_window": (52, 152), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5)},
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

        # --- GEÄNDERT: Low Count (i=0) wird nie gefittet ---
        fit_y, par, perr = (None, None, None)
        if i > 0:
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

        # 2. Raw Intensitäten (GEÄNDERT: Explizites y_lim_raw aus Config)
        ax2 = axes[1, i]
        ax2.plot(x_axis, results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(x_axis, results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(cfg["fit_window"][0], cfg["fit_window"][1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(cfg.get("y_lim_raw", (1.5, 6))) 
        if i==0: ax2.set_ylabel("Counts")
        if i==1: ax2.legend(loc='upper right', fontsize=8)

        # 3. SRBR + Gaussian Fit (GEÄNDERT: Explizites y_lim_sbr aus Config)
        ax3 = axes[2, i]
        ax3.errorbar(x_axis, results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, "
                 f"σ={p[2]:.2f}±{e[2]:.2f}, Max={np.max(results[i]['fit']):.2f})")
            ax3.plot(x_axis, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
        ax3.set_xlim(cfg["fit_window"])
        ax3.set_ylim(cfg.get("y_lim_sbr", (-0.1, 0.5)))
        ax3.set_xlabel("Pixel X")
        ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    save_dir = OUT_DIR / f"Serie_{s_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"Analysis_L_{rank_name}_S{s_id}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. Main & Checksumme
# =====================================================
def main():
    matplotlib.use("Agg")
    for rank_name, model_file in MODELS.items():
        for s_id in sorted(SERIES_CONFIG.keys()):
            cfg = SERIES_CONFIG[s_id]
            process_combination(rank_name, s_id, cfg)
            
    print("\nFertig! Alle L-Plots erstellt.")

if __name__ == "__main__":
    main()