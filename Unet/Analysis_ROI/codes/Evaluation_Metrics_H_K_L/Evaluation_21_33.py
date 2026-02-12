import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE (Korrigiert laut Bild)
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")

# Korrektur: Der Ordner heißt "Prediction.npz" und liegt direkt in "Unet"
IN_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "Prediction.npz" / "Predictions_Raw_21_33"

LOG_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "codes" / "Evaluation_Metrics_H_K_L"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = LOG_DIR / "CDW_Hyperparameter_Results_P0_P1.csv"

# Die 18 Modelle für P0 und P1
NEW_MODELS = {
    # --- P0 ---
    "P0_Seed43": "InfSeed_P0_a0.0000_b0.0000_seed43_20260210-170149_loss0.0195_val0.0224.keras",
    "P0_Seed44": "InfSeed_P0_a0.0000_b0.0000_seed44_20260210-180919_loss0.0195_val0.0224.keras",
    "P0_Seed47": "InfSeed_P0_a0.0000_b0.0000_seed47_20260210-193132_loss0.0158_val0.0181.keras",
    "P0_Seed50": "InfSeed_P0_a0.0000_b0.0000_seed50_20260210-204618_loss0.0191_val0.0219.keras",
    "P0_Seed62": "InfSeed_P0_a0.0000_b0.0000_seed62_20260211-001540_loss0.0152_val0.0180.keras",
    "P0_Seed63": "InfSeed_P0_a0.0000_b0.0000_seed63_20260211-013023_loss0.0155_val0.0180.keras",
    "P0_Seed65": "InfSeed_P0_a0.0000_b0.0000_seed65_20260211-024800_loss0.0154_val0.0182.keras",
    "P0_Seed69": "InfSeed_P0_a0.0000_b0.0000_seed69_20260212-092553_loss0.0161_val0.0182.keras",
    "P0_Seed75": "InfSeed_P0_a0.0000_b0.0000_seed75_20260212-110809_loss0.0203_val0.0226.keras",

    # --- P1 ---
    "P1_Seed43": "InfSeed_P1_a0.8333_b0.0000_seed43_20260210-170150_loss0.0662_val0.0747.keras",
    "P1_Seed44": "InfSeed_P1_a0.8333_b0.0000_seed44_20260210-180249_loss0.0655_val0.0741.keras",
    "P1_Seed45": "InfSeed_P1_a0.8333_b0.0000_seed45_20260210-190307_loss0.0655_val0.0746.keras",
    "P1_Seed46": "InfSeed_P1_a0.8333_b0.0000_seed46_20260210-200941_loss0.0659_val0.0745.keras",
    "P1_Seed47": "InfSeed_P1_a0.8333_b0.0000_seed47_20260210-211638_loss0.0662_val0.0744.keras",
    "P1_Seed48": "InfSeed_P1_a0.8333_b0.0000_seed48_20260210-222216_loss0.0658_val0.0751.keras",
    "P1_Seed49": "InfSeed_P1_a0.8333_b0.0000_seed49_20260210-233303_loss0.0661_val0.0744.keras",
    "P1_Seed50": "InfSeed_P1_a0.8333_b0.0000_seed50_20260211-003034_loss0.0663_val0.0752.keras",
    "P1_Seed53": "InfSeed_P1_a0.8333_b0.0000_seed53_20260211-020101_loss0.0658_val0.0742.keras",
}

# --- Konfiguration für Serien (unverändert) ---
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

# --- Hilfsfunktionen für Fitting ---
def gaussian(x, A, mu, sigma): return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_fit(x, y, win):
    mask = (x >= win[0]) & (x <= win[1])
    xf, yf = x[mask], y[mask]
    if len(yf) < 5 or np.max(yf) <= 0: return None
    p0 = [np.max(yf), xf[np.argmax(yf)], 4.0]
    lb = [0, win[0] - 10, 0.5]; ub = [np.inf, win[1] + 10, 20.0]
    try:
        popt, _ = curve_fit(gaussian, xf, yf, p0=p0, bounds=(lb, ub), maxfev=3000)
        return popt
    except: return None

def get_stable_sbr(sig_vals, bg_vals):
    stable_bg = max(np.mean(bg_vals), 1e-4)
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
    else: # L
        sig = np.mean(img[y1:y2, 0:240], axis=0)
        bg  = np.mean(img[max(0, y1-15):max(0, y1-5), 0:240], axis=0)
        return np.arange(240), get_stable_sbr(sig, bg)

# =====================================================
# 3. EVALUATION START
# =====================================================
def run_evaluation():
    all_results = []
    
    for rank_key, full_keras_name in NEW_MODELS.items():
        match = re.search(r'InfSeed_(P\d)_a(\d+\.\d+)_b(\d+\.\d+)_seed(\d+)', full_keras_name)
        if not match: continue
        
        p_type, alpha, beta, seed = match.group(1), float(match.group(2)), float(match.group(3)), match.group(4)
        print(f">>> Processing {rank_key}...")

        for s_id, cfg in SERIES_CONFIG.items():
            # Dateiname wie von deinem ersten Skript erzeugt: Pred_P0_Seed43_D5_S5_FullSeries.npz
            path = IN_DIR / f"Pred_{rank_key}_D5_S{s_id}_FullSeries.npz"
            
            if not path.exists():
                continue
            
            data = np.load(path)
            v_pr, v_gt = np.clip(data['pred'], 0, None), np.clip(data['gt'], 0, None)

            for d in ["H", "K", "L"]:
                cur_cfg = cfg.copy()
                if d == "H" and s_id == 35: cur_cfg["roi_x"] = (128, 149)
                
                x, p_pr = extract_profile(v_pr, cur_cfg, d)
                _, p_gt = extract_profile(v_gt, cur_cfg, d)
                win = cur_cfg[f"win_{d.lower()}"]

                f_p, f_g = perform_fit(x, p_pr, win), perform_fit(x, p_gt, win)
                
                if f_p is not None and f_g is not None and f_g[0] > 0.01:
                    all_results.append({
                        "Type": p_type, "Seed": seed, "Alpha": alpha, "Beta": beta, "Dir": d,
                        "AreaRatio": (f_p[0]*f_p[2])/(f_g[0]*f_g[2]),
                        "SBRGain": f_p[0]/f_g[0],
                        "PosShift": abs(f_p[1]-f_g[1])
                    })

    if not all_results:
        print(f"Keine Dateien gefunden in: {IN_DIR}")
        return

    df = pd.DataFrame(all_results)
    
    # Mittelwert über H, K, L pro Run
    summary = df.groupby(["Type", "Seed", "Alpha", "Beta"]).agg({
        "AreaRatio": "mean", "SBRGain": "mean", "PosShift": "mean"
    }).reset_index()

    # Sortierung und Trennung
    df_p0 = summary[summary["Type"] == "P0"].sort_values("Seed")
    df_p1 = summary[summary["Type"] == "P1"].sort_values("Seed")

    # Speichern mit manuellem Abstand
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# --- SUCCESS POINT 0 (P0) ---\n")
        df_p0.to_csv(f, index=False)
        f.write("\n\n") # Der gewünschte Abstand
        f.write("# --- SUCCESS POINT 1 (P1) ---\n")
        df_p1.to_csv(f, index=False, header=True) # Header für P1 zur besseren Lesbarkeit

    print(f"\nFertig! CSV gespeichert unter:\n{OUT_FILE}")

if __name__ == "__main__":
    run_evaluation()