#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

# Mathematische Warnungen unterdrücken
warnings.filterwarnings("ignore")

# =====================================================
# 1. PFADE & SETUP
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Unet/Analysis_ROI/Predictions_Raw"
OUT_FILE = ROOT_DIR / "Evaluation_Metrics_Summary.txt"

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
    5:  {"slice_idx": 15, "roi_x": (195, 216), "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (140,240)},
    11: {"slice_idx": 20, "roi_x": (76, 97),   "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (43,143)},
    12: {"slice_idx": 18, "roi_x": (60, 81),   "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (24,124)},
    15: {"slice_idx": 19, "roi_x": (136, 157), "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (98,198)},
    16: {"slice_idx": 17, "roi_x": (115, 136), "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (76,176)},
    21: {"slice_idx": 19, "roi_x": (192, 213), "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (140,240)},
    22: {"slice_idx": 17, "roi_x": (176, 197), "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (134,234)},
    29: {"slice_idx": 25, "roi_x": (50, 71),   "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (20,120)},
    35: {"slice_idx": 24, "roi_x": (98, 119),  "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (64,164)},
    50: {"slice_idx": 13, "roi_x": (92, 113),  "roi_y": (102, 117), "bg_gap": 5, "fH": (2,38), "fK": (90,130), "fL": (52,152)},
}

FIX_W, FIX_H = 21, 11
BG_BOX_HEIGHT = 10

# =====================================================
# 2. MATHEMATIK & FIT-LOGIK
# =====================================================
def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    xf, yf = x[mask], y[mask]
    if len(yf) < 5: return None
    norm = np.max(yf) if np.max(yf) > 0 else 1
    yf_n = yf / norm
    p0 = [1.0, xf[np.argmax(yf_n)], 5.0]
    try:
        popt, _ = curve_fit(gaussian, xf, yf_n, p0=p0, maxfev=5000, 
                            bounds=([0, fit_window[0], 0.5], [np.inf, fit_window[1], 30.0]))
        popt[0] *= norm
        res = yf - gaussian(xf, *popt)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((yf - np.mean(yf))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0
        return {"amp": popt[0], "mu": popt[1], "sigma": popt[2], "r2": r2}
    except: return None

def get_sigma_bg(img, cfg, direction):
    x1, x2 = cfg["roi_x"]; y1, y2 = cfg["roi_y"]
    if direction in ["H", "L"]:
        ty2 = max(0, y1 - cfg["bg_gap"]); ty1 = max(0, ty2 - BG_BOX_HEIGHT)
        by1 = min(192, y2 + cfg["bg_gap"]); by2 = min(192, by1 + BG_BOX_HEIGHT)
        bg_px = np.concatenate([img[ty1:ty2, x1:x2].flatten(), img[by1:by2, x1:x2].flatten()])
    else: # K
        bl = min(240, x2 + cfg["bg_gap"]); br = min(240, bl + 20)
        bg_px = img[y1:y2, bl:br].flatten()
    return np.std(bg_px) if bg_px.size > 0 else 1e-6

# Zentrierung
def update_config(cfg_dict, w, h):
    new_cfg = {}
    for s_id, vals in cfg_dict.items():
        v = vals.copy()
        cx = (v["roi_x"][0] + v["roi_x"][1]) / 2
        v["roi_x"] = (int(cx - w//2), int(cx - w//2 + w))
        cy = (v["roi_y"][0] + v["roi_y"][1]) / 2
        v["roi_y"] = (int(cy - h//2), int(cy - h//2 + h))
        new_cfg[s_id] = v
    return new_cfg

SERIES_CONFIG = update_config(SERIES_CONFIG, FIX_W, FIX_H)

# =====================================================
# 3. EVALUATIONS-KERN
# =====================================================
def evaluate_run(rank_label):
    m_dr2, m_snr, m_int, m_red, m_dmu = [], [], [], [], []
    jit = {"H": {"g":[], "p":[]}, "K": {"g":[], "p":[]}, "L": {"g":[], "p":[]}}
    sum_area_gt, sum_area_pred = 0, 0

    for s_id, cfg in SERIES_CONFIG.items():
        path = IN_DIR / f"Pred_{rank_label}_D5_S{s_id}_FullSeries.npz"
        if not path.exists(): continue
        data = np.load(path)

        for d in ["H", "K", "L"]:
            if d == "H":
                x = np.arange(41)
                p_g = np.array([np.sum(img[cfg['roi_y'][0]:cfg['roi_y'][1], cfg['roi_x'][0]:cfg['roi_x'][1]]) for img in data['gt']])
                p_p = np.array([np.sum(img[cfg['roi_y'][0]:cfg['roi_y'][1], cfg['roi_x'][0]:cfg['roi_x'][1]]) for img in data['pred']])
                sig_g = np.mean([get_sigma_bg(img, cfg, d) for img in data['gt']])
                sig_p = np.mean([get_sigma_bg(img, cfg, d) for img in data['pred']])
                win = cfg["fH"]
            else:
                idx = cfg["slice_idx"]
                img_g, img_p = data['gt'][idx], data['pred'][idx]
                sig_g, sig_p = get_sigma_bg(img_g, cfg, d), get_sigma_bg(img_p, cfg, d)
                if d == "K":
                    x = np.arange(192); win = cfg["fK"]
                    p_g = np.sum(img_g[:, cfg['roi_x'][0]:cfg['roi_x'][1]], axis=1)
                    p_p = np.sum(img_p[:, cfg['roi_x'][0]:cfg['roi_x'][1]], axis=1)
                else: # L
                    x = np.arange(240); win = cfg["fL"]
                    p_g = np.sum(img_g[cfg['roi_y'][0]:cfg['roi_y'][1], :], axis=0)
                    p_p = np.sum(img_p[cfg['roi_y'][0]:cfg['roi_y'][1], :], axis=0)

            f_g, f_p = perform_fit(x, p_g, np.full_like(x, sig_g), win), perform_fit(x, p_p, np.full_like(x, sig_p), win)

            if f_g and f_p:
                m_dr2.append(f_p["r2"] - f_g["r2"])
                m_snr.append(20 * np.log10((f_p["amp"]/sig_p) / (f_g["amp"]/sig_g)))
                m_int.append((f_p["amp"]*f_p["sigma"]) / (f_g["amp"]*f_g["sigma"]))
                m_red.append(sig_g / sig_p)
                # NEU: Absolute Abweichung der Peak-Position
                m_dmu.append(abs(f_p["mu"] - f_g["mu"]))
                
                jit[d]["g"].append(f_g["mu"]); jit[d]["p"].append(f_p["mu"])
                sum_area_gt += (f_g["amp"] * f_g["sigma"])
                sum_area_pred += (f_p["amp"] * f_p["sigma"])

    jitter_vals = [np.std(jit[d]["g"])/np.std(jit[d]["p"]) for d in jit if len(jit[d]["g"]) > 1 and np.std(jit[d]["p"]) > 0]
    
    return {
        "dr2": np.mean(m_dr2) if m_dr2 else 0,
        "snr": np.mean(m_snr) if m_snr else 0,
        "int": np.median(m_int) if m_int else 0,
        "red": np.mean(m_red) if m_red else 0,
        "jit": np.mean(jitter_vals) if jitter_vals else 1.0,
        "tii": sum_area_pred / sum_area_gt if sum_area_gt > 0 else 0,
        "dmu": np.mean(m_dmu) if m_dmu else 0 # 7. Metrik
    }

# =====================================================
# 4. MAIN & EXPORT
# =====================================================
def main():
    print(f"Master-Evaluation mit 7 Metriken...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        header = f"{'Model':<12} | {'dR2':>6} | {'SNR[dB]':>8} | {'IntFid':>7} | {'NoiseRed':>8} | {'Jitter':>6} | {'TotInt':>7} | {'dMu':>6}\n"
        f.write(header + "-" * 85 + "\n")
        
        for rank in MODELS.keys():
            print(f"Berechne {rank}...")
            res = evaluate_run(rank)
            line = (f"{rank:<12} | {res['dr2']:>6.3f} | {res['snr']:>8.2f} | "
                    f"{res['int']:>7.2f} | {res['red']:>8.2f} | {res['jit']:>6.2f} | "
                    f"{res['tii']:>7.3f} | {res['dmu']:>6.3f}\n")
            f.write(line)
    print(f"Fertig! Ergebnisse in: {OUT_FILE}")

if __name__ == "__main__":
    main()