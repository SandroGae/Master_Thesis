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
    OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Analysis_H_Direction_RERUN"
else:
    print(">>> Nutze ORIGINAL Setup (K-Richtung)")
    MODELS = MODELS_ORIGINAL
    IN_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw"
    OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Analysis_H_Direction"

OUT_DIR.mkdir(parents=True, exist_ok=True)



# =====================================================
# 2. SERIEN-KONFIGURATION (mit individuellen Y-Achsen)
# =====================================================
SERIES_CONFIG = {
    5:  {"slice_idx": 15, "roi_x": (195, 216), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    11: {"slice_idx": 20, "roi_x": (76, 97),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    15: {"slice_idx": 19, "roi_x": (136, 157), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    16: {"slice_idx": 17, "roi_x": (115, 136), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    22: {"slice_idx": 17, "roi_x": (176, 197), "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    29: {"slice_idx": 25, "roi_x": (50, 71),   "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 99.0), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
    35: {"slice_idx": 24, "roi_x": (98, 119),  "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.0), "ylim_raw": (40, 65),   "ylim_sbr": (-0.1, 0.4)},
    50: {"slice_idx": 13, "roi_x": (92, 113),  "roi_y": (102, 117), "bg_gap": 5, "vis_p": (0.5, 98.5), "ylim_raw": (40, 65), "ylim_sbr": (-0.1, 0.4)},
}

# NEUE FIXE GEOMETRIE
FIX_W, FIX_H = 21, 11
BG_BOX_HEIGHT = 10
FIT_WIN       = (2, 38)
FIT_COLORS    = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES        = ["Low Count", "Prediction", "Ground Truth"]

# =====================================================
# 3. AUTOMATISCHE ZENTRIERUNG & MATHEMATIK
# =====================================================
def update_config_to_fixed_size(cfg_dict, w, h):
    """Zentriert die ROI auf die neue feste Größe (21x11)"""
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
    t_y2 = max(0, y_start - gap)
    t_y1 = max(0, t_y2 - BG_BOX_HEIGHT)
    b_y1 = min(192, y_end + gap)
    b_y2 = min(192, b_y1 + BG_BOX_HEIGHT)
    return (t_y1, t_y2), (b_y1, b_y2)

def calculate_sbr_z_profiles(volume, cfg, bg_coords, force_noise_std_array=None):
    n_frames = volume.shape[0]
    x1, x2 = cfg["roi_x"]
    y1, y2 = cfg["roi_y"]
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
        err_sig = px_std * np.sqrt(n_sig_pixels)
        err_bg = px_std * np.sqrt(bg_concat.size) * scale
        err_net = np.sqrt(err_sig**2 + err_bg**2)
        
        rel_err_net = err_net / max(abs(net_signal), 1e-9)
        rel_err_bg = err_bg / max(sum_bg_equiv, 1e-9)
        total_rel = np.sqrt(rel_err_net**2 + rel_err_bg**2)
        
        p_sig.append(sum_signal); p_bg.append(sum_bg_equiv)
        p_sbr.append(sbr); p_err.append(abs(sbr) * total_rel)

    return np.arange(n_frames), np.array(p_sig), np.array(p_bg), np.array(p_sbr), np.array(p_err)

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
# 4. MAIN LOOP
# =====================================================
def main():
    matplotlib.use('Agg')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rank_label, model_file in MODELS.items():
        print(f"Processing Model: {rank_label}")
        for s_id, cfg in SERIES_CONFIG.items():
            file_path = IN_DIR / f"Pred_{rank_label}_D5_S{s_id}_FullSeries.npz"
            if not file_path.exists(): continue

            series_dir = OUT_DIR / f"Serie_{s_id}"
            series_dir.mkdir(parents=True, exist_ok=True)

            data = np.load(file_path)
            volumes = [data['lc'], data['pred'], data['gt']]
            bg_coords = get_bg_coords_vertical(cfg)
            
            gt_noise = []
            for v in volumes[2]:
                (ty1, ty2), (by1, by2) = bg_coords
                bg_list = []
                if ty2 > ty1: bg_list.append(v[ty1:ty2, cfg["roi_x"][0]:cfg["roi_x"][1]])
                if by2 > by1: bg_list.append(v[by1:by2, cfg["roi_x"][0]:cfg["roi_x"][1]])
                gt_noise.append(np.std(np.concatenate(bg_list)))

            results = []
            for i, vol in enumerate(volumes):
                noise = gt_noise if i == 1 else None
                x, sig, bg, sbr, err = calculate_sbr_z_profiles(vol, cfg, bg_coords, noise)
                
                # GEÄNDERT: Nur Prediction (1) und GT (2) fitten, Low Count (0) nie.
                fit_y, par, perr = (None, None, None)
                if i > 0:
                    fit_y, par, perr = perform_gaussian_fit(x, sbr, err, FIT_WIN)
                
                results.append({'x':x, 'sig':sig, 'bg':bg, 'sbr':sbr, 'err':err, 'fit':fit_y, 'par':par, 'perr':perr})

            fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
            p_low, p_high = cfg["vis_p"]

            for i in range(3):
                # ZEILE 0: Bilder
                ax = axes[0, i]
                ax.imshow(vis_norm(volumes[i][cfg["slice_idx"]], p_low, p_high), cmap="gray_r")
                ax.set_title(f"{TITLES[i]} (S{s_id})", fontsize=14, fontweight='bold')
                
                x1, x2 = cfg["roi_x"]; y1, y2 = cfg["roi_y"]
                rw, rh = x2 - x1, y2 - y1
                ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=0, fc='green', alpha=0.3))
                ax.add_patch(patches.Rectangle((x1, y1), rw, rh, lw=2, ec='blue', fc='none'))
                (ty1, ty2), (by1, by2) = bg_coords
                if ty2 > ty1: ax.add_patch(patches.Rectangle((x1, ty1), rw, ty2-ty1, lw=1, ec='red', fc='red', alpha=0.2))
                if by2 > by1: ax.add_patch(patches.Rectangle((x1, by1), rw, by2-by1, lw=1, ec='red', fc='red', alpha=0.2))
                ax.axis('off')

                # ZEILE 1: Raw Intensitäten (mit individuellem Y-Limit)
                ax2 = axes[1, i]
                ax2.plot(results[i]['x'], results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
                ax2.plot(results[i]['x'], results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
                ax2.axvspan(FIT_WIN[0], FIT_WIN[1], color='green', alpha=0.15, label='_Fit Region')
                ax2.grid(True, alpha=0.3)
                if i==0: ax2.set_ylabel("Counts")
                if i==1: ax2.legend(loc='upper right', fontsize=8)
                
                # Y-Achse Logik
                y_lim_config = cfg.get("ylim_raw", (None, None))
                if y_lim_config[0] is None:
                    y_min, y_max = np.min(results[i]['sig']), np.max(results[i]['sig'])
                    ax2.set_ylim(y_min*0.9, y_max*1.1)
                else:
                    ax2.set_ylim(y_lim_config)

                # ZEILE 2: SRBR & Fit (mit individuellem Y-Limit)
                ax3 = axes[2, i]
                ax3.errorbar(results[i]['x'], results[i]['sbr'], yerr=results[i]['err'], fmt='.', color='black', alpha=0.6, label='SRBR')
                ax3.axhline(0, color='gray', ls=':', alpha=0.5)
                if results[i]['fit'] is not None:
                    p, e = results[i]['par'], results[i]['perr']
                    l = (f"Gauss (Peak={p[1]:.1f}±{e[1]:.1f}, σ={p[2]:.2f}±{e[2]:.2f})")
                    ax3.plot(results[i]['x'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
                
                ax3.set_xlabel("Image Index (Z-Axis)")
                if i==0: ax3.set_ylabel("SRBR")
                ax3.grid(True, alpha=0.3); ax3.legend(loc='upper right', fontsize=8)
                ax3.set_ylim(cfg.get("ylim_sbr", (-0.1, 0.55)))

            plt.tight_layout()
            out_name = f"Ana_H_Dir_{rank_label}_S{s_id}.png"
            fig.savefig(series_dir / out_name)
            plt.close(fig)

    print(f"\nFertig! Alle Plots in {OUT_DIR} gespeichert.")

if __name__ == "__main__":
    main()