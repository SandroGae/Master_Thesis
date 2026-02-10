import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR = ROOT_DIR / "Unet/Analysis_ROI/Prediction.npz/Predictions_Raw_new_RERUN"
LOG_DIR = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = LOG_DIR / "CDW_Hyperparameter_Results.csv"

# Stufen für das 7x7 Raster
STEPS = [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 1.0]

# Vollständige Liste deiner 43 Modelle (NEW_RERUN)
KERAS_MODELS = {
    "Rang_01": "Rang_01_DeepScan_a0.1667_b0.0000_seed42_20260131-011255_loss0.0257_val0.0294.keras",
    "Rang_02": "Rang_02_DeepScan_a0.3333_b0.5000_seed42_20260131-022758_loss0.0839_val0.0717.keras",
    "Rang_03": "Rang_03_DeepScan_a0.3333_b0.0000_seed42_20260131-021835_loss0.0361_val0.0407.keras",
    "Rang_04": "Rang_04_DeepScan_a0.3333_b0.3333_seed42_20260131-010926_loss0.0657_val0.0614.keras",
    "Rang_05": "Rang_05_DeepScan_a0.1667_b0.3333_seed43_20260131-002251_loss0.0582_val0.0554.keras",
    "Rang_06": "Rang_06_DeepScan_a0.3333_b0.1667_seed42_20260131-000951_loss0.0533_val0.0511.keras",
    "Rang_07": "Rang_07_DeepScan_a0.1667_b0.5000_seed42_20260131-014158_loss0.0744_val0.0684.keras",
    "Rang_08": "Rang_08_DeepScan_a0.5000_b0.1667_seed42_20260131-011728_loss0.0574_val0.0601.keras",
    "Rang_09": "Rang_09_DeepScan_a0.1667_b0.1667_seed42_20260131-030855_loss0.0479_val0.0425.keras",
    "Rang_10": "Rang_10_DeepScan_a0.5000_b0.0000_seed42_20260131-001704_loss0.0463_val0.0521.keras",
    "Rang_11": "Rang_11_DeepScan_a0.5000_b0.3333_seed42_20260131-031358_loss0.0683_val0.0678.keras",
    "Rang_12": "Rang_12_DeepScan_a0.0000_b0.1667_seed42_20260131-021040_loss0.0401_val0.0337.keras",
    "Rang_13": "Rang_13_DeepScan_a0.1667_b0.6667_seed42_20260131-023652_loss0.1009_val0.0814.keras",
    "Rang_14": "Rang_14_DeepScan_a0.3333_b0.6667_seed42_20260131-001041_loss0.0879_val0.0822.keras",
    "Rang_15": "Rang_15_DeepScan_a0.1667_b0.8333_seed42_20260131-000640_loss0.1355_val0.0945.keras",
    "Rang_16": "Rang_16_DeepScan_a0.5000_b0.5000_seed42_20260131-002205_loss0.0756_val0.0753.keras",
    "Rang_17": "Rang_17_DeepScan_a0.6667_b0.1667_seed42_20260131-053158_loss0.0631_val0.0685.keras",
    "Rang_18": "Rang_18_DeepScan_a0.3333_b0.8333_seed42_20260131-010607_loss0.1229_val0.0927.keras",
    "Rang_19": "Rang_19_DeepScan_a0.5000_b0.6667_seed42_20260131-011741_loss0.0923_val0.0830.keras",
    "Rang_20": "Rang_20_DeepScan_a0.6667_b0.3333_seed43_20260131-032304_loss0.0723_val0.0736.keras",
    "Rang_21": "Rang_21_DeepScan_a0.6667_b0.5000_seed42_20260131-042251_loss0.0825_val0.0788.keras",
    "Rang_22": "Rang_22_DeepScan_a0.6667_b0.0000_seed42_20260131-044258_loss0.0577_val0.0638.keras",
    "Rang_23": "Rang_23_DeepScan_a0.5000_b1.0000_seed43_20260131-031943_loss0.0993_val0.0984.keras",
    "Rang_24": "Rang_24_DeepScan_a0.5000_b0.8333_seed43_20260131-024341_loss0.1022_val0.0910.keras",
    "Rang_25": "Rang_25_DeepScan_a0.8333_b0.0000_seed42_20260131-061901_loss0.0662_val0.0746.keras",
    "Rang_26": "Rang_26_DeepScan_a0.6667_b0.6667_seed44_20260131-055426_loss0.0887_val0.0842.keras",
    "Rang_27": "Rang_27_DeepScan_a0.3333_b1.0000_seed42_20260131-020559_loss0.1302_val0.1031.keras",
    "Rang_28": "Rang_28_DeepScan_a0.8333_b0.1667_seed43_20260131-041442_loss0.0707_val0.0773.keras",
    "Rang_29": "Rang_29_DeepScan_a0.8333_b0.6667_seed43_20260131-043624_loss0.0821_val0.0851.keras",
    "Rang_30": "Rang_30_DeepScan_a0.8333_b0.3333_seed43_20260131-052558_loss0.0741_val0.0801.keras",
    "Rang_31": "Rang_31_DeepScan_a0.6667_b0.8333_seed42_20260131-033152_loss0.0990_val0.0892.keras",
    "Rang_32": "Rang_32_DeepScan_a0.1667_b1.0000_seed42_20260131-012331_loss0.1229_val0.1074.keras",
    "Rang_33": "Rang_33_DeepScan_a0.0000_b0.0000_seed43_20260131-003040_loss0.0152_val0.0184.keras",
    "Rang_34": "Rang_34_DeepScan_a0.8333_b0.5000_seed42_20260131-061643_loss0.0791_val0.0824.keras",
    "Rang_35": "Rang_35_DeepScan_a0.8333_b0.8333_seed42_20260131-052931_loss0.0845_val0.0878.keras",
    "Rang_36": "Rang_36_DeepScan_a0.6667_b1.0000_seed42_20260131-052908_loss0.1067_val0.0946.keras",
    "Rang_37": "Rang_37_DeepScan_a0.8333_b1.0000_seed43_20260131-064157_loss0.0893_val0.0904.keras",
    "Rang_38": "Rang_38_DeepScan_a1.0000_b0.0000_seed42_20260131-073210_loss0.0766_val0.0862.keras",
    "Rang_39": "Rang_39_DeepScan_a0.0000_b0.5000_seed42_20260131-000641_loss0.0809_val0.0671.keras",
    "Rang_40": "Rang_40_DeepScan_a0.0000_b0.3333_seed42_20260131-040643_loss0.0541_val0.0522.keras",
    "Rang_41": "Rang_41_DeepScan_a0.0000_b0.6667_seed42_20260131-015925_loss0.1080_val0.0820.keras",
    "Rang_42": "Rang_42_DeepScan_a0.0000_b0.8333_seed42_20260131-034436_loss0.1071_val0.0969.keras",
    "Rang_43": "Rang_43_DeepScan_a0.0000_b1.0000_seed42_20260131-000640_loss0.1372_val0.1118.keras",
}

SERIES_CONFIG = {
    5:  {"z_idx": 15, "roi_x": (195, 216), "roi_y": (102, 117), "bg_gap_k": -211, "win_h": (2,38), "win_k": (90,130), "win_l": (140,240)},
    11: {"z_idx": 20, "roi_x": (76, 97),   "roi_y": (102, 117), "bg_gap_k": 52,   "win_h": (2,38), "win_k": (90,130), "win_l": (43,143)},
    12: {"z_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap_k": 64,   "win_h": (2,38), "win_k": (90,130), "win_l": (24,124)},
    15: {"z_idx": 19, "roi_x": (136, 157), "roi_y": (102, 117), "bg_gap_k": -162, "win_h": (2,38), "win_k": (90,130), "win_l": (98,198)},
    16: {"z_idx": 17, "roi_x": (115, 136), "roi_y": (102, 117), "bg_gap_k": 78,   "win_h": (2,38), "win_k": (90,130), "win_l": (76,176)},
    21: {"z_idx": 19, "roi_x": (192, 213), "roi_y": (102, 117), "bg_gap_k": -213, "win_h": (2,38), "win_k": (90,130), "win_l": (140,240)},
    22: {"z_idx": 17, "roi_x": (176, 197), "roi_y": (102, 117), "bg_gap_k": -183, "win_h": (2,38), "win_k": (90,130), "win_l": (134,234)},
    29: {"z_idx": 25, "roi_x": (50, 71),   "roi_y": (102, 117), "bg_gap_k": -80,  "win_h": (2,38), "win_k": (90,130), "win_l": (20,120)},
    35: {"z_idx": 24, "roi_x": (106, 127), "roi_y": (102, 117), "bg_gap_k": 45,   "win_h": (2,38), "win_k": (90,130), "win_l": (64,164)},
    50: {"z_idx": 13, "roi_x": (92, 113),  "roi_y": (102, 117), "bg_gap_k": 38,   "win_h": (2,38), "win_k": (90,130), "win_l": (52,152)},
}

# =====================================================
# 2. STABILE MATHEMATIK
# =====================================================
def gaussian(x, A, mu, sigma): return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_fit(x, y, win):
    mask = (x >= win[0]) & (x <= win[1])
    xf, yf = x[mask], y[mask]
    if len(yf) < 5 or np.max(yf) <= 0: return None
    p0 = [np.max(yf), xf[np.argmax(yf)], 4.0]
    # PHYSIKALISCHE SCHRANKEN (lb = lower bound, ub = upper bound)
    lb = [0, win[0] - 10, 0.5]
    ub = [np.inf, win[1] + 10, 20.0]
    try:
        popt, _ = curve_fit(gaussian, xf, yf, p0=p0, bounds=(lb, ub), maxfev=3000)
        return popt
    except: return None

def get_stable_sbr(sig_vals, bg_vals):
    # Mittelwert der Noise-Pixel
    mean_bg = np.mean(bg_vals)
    stable_bg = max(mean_bg, 1e-4) # Div-by-Zero Schutz
    return (sig_vals - stable_bg) / stable_bg

def extract_profile(vol, cfg, direction):
    y1, y2, x1, x2 = cfg["roi_y"][0], cfg["roi_y"][1], cfg["roi_x"][0], cfg["roi_x"][1]
    if direction == "H":
        sig = np.mean(vol[:, y1:y2, x1:x2], axis=(1, 2))
        bg  = np.mean(vol[:, max(0, y1-15):max(0, y1-5), x1:x2], axis=(1, 2))
        return np.arange(vol.shape[0]), get_stable_sbr(sig, bg)
    img = vol[cfg["z_idx"]]
    if direction == "K":
        sig = np.mean(img[0:192, x1:x2], axis=1)
        bg_l = max(0, min(239, x2 + cfg["bg_gap_k"]))
        bg  = np.mean(img[0:192, bg_l:min(240, bg_l+10)], axis=1)
        return np.arange(192), get_stable_sbr(sig, bg)
    else: # L-Richtung
        sig = np.mean(img[y1:y2, 0:240], axis=0)
        bg  = np.mean(img[max(0, y1-15):max(0, y1-5), 0:240], axis=0)
        return np.arange(240), get_stable_sbr(sig, bg)

# =====================================================
# 3. EVALUATION
# =====================================================
def run_evaluation():
    all_results = []
    rejected_count = 0
    
    for rank_key, full_keras_name in KERAS_MODELS.items():
        # Match hyperparameter string (z.B. DeepScan_a0.1667_b0.0000_seed42)
        match_params = re.search(r'Rang_\d+_(.*)_2026', full_keras_name)
        if not match_params: continue
        file_id = match_params.group(1)
        
        # Alpha/Beta für CSV
        alpha_raw = float(re.search(r'a(\d+\.\d+)', full_keras_name).group(1))
        beta_raw = float(re.search(r'b(\d+\.\d+)', full_keras_name).group(1))
        
        print(f"\n>>> Evaluiere {rank_key}: a={alpha_raw:.4f}, b={beta_raw:.4f}")

        for s_id, cfg in SERIES_CONFIG.items():
            # Dateisuche
            path = IN_DIR / f"Pred_{file_id}_D5_S{s_id}_FullSeries.npz"
            if not path.exists(): continue
            
            data = np.load(path)
            v_pr, v_gt = np.clip(data['pred'], 0, None), np.clip(data['gt'], 0, None)

            for d in ["H", "K", "L"]:
                cur_cfg = cfg.copy()
                if d == "H" and s_id == 35: cur_cfg["roi_x"] = (128, 149) # S35 H-Fix
                
                x, p_pr = extract_profile(v_pr, cur_cfg, d)
                _, p_gt = extract_profile(v_gt, cur_cfg, d)
                win = cur_cfg[f"win_{d.lower()}"]

                f_p = perform_fit(x, p_pr, win)
                f_g = perform_fit(x, p_gt, win)
                
                if f_p is not None and f_g is not None and f_g[0] > 0.01:
                    all_results.append({
                        "Alpha": alpha_raw, "Beta": beta_raw, "Dir": d,
                        "AreaRatio": (f_p[0]*f_p[2])/(f_g[0]*f_g[2]),
                        "SBRGain": f_p[0]/f_g[0],
                        "PosShift": abs(f_p[1]-f_g[1])
                    })
                else:
                    rejected_count += 1
                    reason = "Fit failed/20px bound" if f_p is None else "GT too weak"
                    print(f"    [!] Rejected: S{s_id} {d}-Dir ({reason})")

    if not all_results:
        print("Keine Daten gefunden!")
        return

    df = pd.DataFrame(all_results)
    # Runden auf striktes 7x7 Gitter
    df["Alpha"] = df["Alpha"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    df["Beta"] = df["Beta"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    
    # Aggregation inkl. Anzahl der erfolgreichen Fits ("Valid_Fits")
    summary = df.groupby(["Alpha", "Beta", "Dir"]).agg({
        "AreaRatio": "mean",
        "SBRGain": "mean",
        "PosShift": "mean",
        "Dir": "count"
    }).rename(columns={"Dir": "Valid_Fits"}).reset_index()

    # ALPHA = 1.0 EXPANSION (Mathematische Äquivalenz)
    a1_data = summary[summary["Alpha"] == 1.0].copy()
    if not a1_data.empty:
        summary = summary[summary["Alpha"] != 1.0]
        for b in STEPS:
            temp = a1_data.copy(); temp["Beta"] = b
            summary = pd.concat([summary, temp])

    summary.to_csv(OUT_FILE, index=False)
    print(f"\n--- ERFOLG ---")
    print(f"Abgelehnte Einzelfits insgesamt: {rejected_count}")
    print(f"Ergebnisse in {OUT_FILE} gespeichert.")

if __name__ == "__main__":
    run_evaluation()