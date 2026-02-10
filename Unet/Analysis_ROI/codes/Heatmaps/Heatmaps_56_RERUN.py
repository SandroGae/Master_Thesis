import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from pathlib import Path
from matplotlib.colors import LightSource
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. PFADE & SETUP
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw_RERUN"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Evaluation_RERUN_Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping bleibt gleich
MODEL_MAP = {
    "Rank_1": (0.1667, 0.0),    "Rank_2": (0.25, 0.0),      "Rank_3": (0.1667, 0.1667),
    "Rank_4": (0.25, 0.5),      "Rank_5": (0.25, 0.0833),   "Rank_6": (0.25, 0.3333),
    "Rank_7": (0.25, 0.25),     "Rank_8": (0.25, 0.1667),   "Rank_9": (0.3333, 0.1667),
    "Rank_10": (0.1667, 0.3333),"Rank_11": (0.3333, 0.3333),"Rank_12": (0.5, 0.0),
    "Rank_13": (0.3333, 0.0),   "Rank_14": (0.1667, 0.5),    "Rank_15": (0.25, 0.4167),
    "Rank_16": (0.25, 0.5833),  "Rank_17": (0.3333, 0.5),    "Rank_18": (0.1667, 0.6667),
    "Rank_19": (0.25, 0.6667),  "Rank_20": (0.5, 0.1667),   "Rank_21": (0.3333, 0.6667),
    "Rank_22": (0.6667, 0.0),   "Rank_23": (0.6667, 0.1667),"Rank_24": (0.5, 0.3333),
    "Rank_25": (0.3333, 0.8333),"Rank_26": (0.5, 0.5),      "Rank_27": (0.25, 0.75),
    "Rank_28": (0.5, 0.6667),   "Rank_29": (0.1667, 0.8333),"Rank_30": (0.25, 0.8333),
    "Rank_31": (0.5, 0.8333),   "Rank_32": (0.6667, 0.3333),"Rank_33": (0.25, 0.9167),
    "Rank_34": (0.8333, 0.0),   "Rank_35": (0.6667, 0.6667),"Rank_36": (0.6667, 0.5),
    "Rank_37": (0.8333, 0.6667),"Rank_38": (0.8333, 0.3333),"Rank_39": (0.6667, 0.8333),
    "Rank_40": (0.8333, 0.5),    "Rank_41": (0.3333, 1.0),   "Rank_42": (0.1667, 1.0),
    "Rank_43": (0.25, 1.0),     "Rank_44": (0.8333, 0.1667),"Rank_45": (0.8333, 0.8333),
    "Rank_46": (0.6667, 1.0),   "Rank_47": (0.5, 1.0),      "Rank_48": (0.8333, 1.0),
    "Rank_49": (1.0, 0.0),      "Rank_50": (0.0, 0.0),      "Rank_51": (0.0, 0.1667),
    "Rank_52": (0.0, 0.3333),   "Rank_53": (0.0, 0.5),      "Rank_54": (0.0, 0.6667),
    "Rank_55": (0.0, 0.8333),   "Rank_56": (0.0, 1.0)
}

# Standard Beta-Stufen für die Replikation
BETA_STEPS = [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 1.0]

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

BG_BOX_HEIGHT = 10

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def gaussian(x, a, mu, sigma): return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_fit(x, y, win):
    norm = np.max(y) if np.max(y) > 0 else 1
    yf_n = y / norm
    mask = (x >= win[0]) & (x <= win[1])
    try:
        popt, _ = curve_fit(gaussian, x[mask], yf_n[mask], p0=[1.0, x[mask][np.argmax(yf_n[mask])], 5.0], 
                            bounds=([0, win[0], 0.5], [np.inf, win[1], 30.0]), maxfev=2000)
        popt[0] *= norm
        res = y[mask] - gaussian(x[mask], *popt)
        r2 = 1 - (np.sum(res**2) / np.sum((y[mask] - np.mean(y[mask]))**2))
        return {"a": popt[0], "mu": popt[1], "sig": popt[2], "r2": r2}
    except: return None

def get_sigma_bg(img, cfg, d):
    x1, x2, y1, y2 = cfg["roi_x"][0], cfg["roi_x"][1], cfg["roi_y"][0], cfg["roi_y"][1]
    if d in ["H", "L"]:
        ty2 = max(0, y1-cfg["bg_gap"]); ty1 = max(0, ty2-BG_BOX_HEIGHT)
        by1 = min(192, y2+cfg["bg_gap"]); by2 = min(192, by1+BG_BOX_HEIGHT)
        bg = np.concatenate([img[ty1:ty2, x1:x2].flatten(), img[by1:by2, x1:x2].flatten()])
    else:
        bl = min(240, x2+cfg["bg_gap"]); br = min(240, bl+20)
        bg = img[y1:y2, bl:br].flatten()
    return np.std(bg) if bg.size > 0 else 1e-6

# =====================================================
# 3. WORKFLOW: METRIKEN BERECHNEN & ALPHA=1 EXPANSION
# =====================================================
def collect_all_metrics():
    all_results = []
    found_files = 0
    
    for rank_key, (alpha, beta) in MODEL_MAP.items():
        rank_num = rank_key.split("_")[1]
        formatted_rank = f"Rank_{int(rank_num):02d}"
        
        for s_id, cfg in SERIES_CONFIG.items():
            path = IN_DIR / f"Pred_{formatted_rank}_D5_S{s_id}_FullSeries.npz"
            if not path.exists(): continue
            
            found_files += 1
            data = np.load(path)
            for d in ["H", "K", "L"]:
                if d == "H":
                    x = np.arange(41); win = cfg["fH"]
                    p_g = np.array([np.sum(img[cfg['roi_y'][0]:cfg['roi_y'][1], cfg['roi_x'][0]:cfg['roi_x'][1]]) for img in data['gt']])
                    p_p = np.array([np.sum(img[cfg['roi_y'][0]:cfg['roi_y'][1], cfg['roi_x'][0]:cfg['roi_x'][1]]) for img in data['pred']])
                    s_g = np.mean([get_sigma_bg(img, cfg, d) for img in data['gt']])
                    s_p = np.mean([get_sigma_bg(img, cfg, d) for img in data['pred']])
                else:
                    idx = cfg["slice_idx"]; x = np.arange(192 if d=="K" else 240); win = cfg["fK"] if d=="K" else cfg["fL"]
                    img_g, img_p = data['gt'][idx], data['pred'][idx]
                    s_g, s_p = get_sigma_bg(img_g, cfg, d), get_sigma_bg(img_p, cfg, d)
                    p_g = np.sum(img_g[:, cfg['roi_x'][0]:cfg['roi_x'][1]], axis=1) if d=="K" else np.sum(img_g[cfg['roi_y'][0]:cfg['roi_y'][1], :], axis=0)
                    p_p = np.sum(img_p[:, cfg['roi_x'][0]:cfg['roi_x'][1]], axis=1) if d=="K" else np.sum(img_p[cfg['roi_y'][0]:cfg['roi_y'][1], :], axis=0)
                
                f_g, f_p = perform_fit(x, p_g, win), perform_fit(x, p_p, win)
                if f_g and f_p:
                    all_results.append({
                        'alpha': alpha, 'beta': beta, 'dir': d, 's_id': s_id,
                        'snr': 20 * np.log10((f_p['a']/s_p)/(f_g['a']/s_g)) if s_p > 0 else 0,
                        'tii': (f_p['a']*f_p['sig'])/(f_g['a']*f_g['sig']),
                        'dmu': abs(f_p['mu'] - f_g['mu']),
                        'mu_g': f_g['mu'], 'mu_p': f_p['mu']
                    })
    
    print(f"\nSuche beendet. {found_files} Dateien geladen.")
    if not all_results: return pd.DataFrame()

    df = pd.DataFrame(all_results)
    
    # --- NEU: ALPHA = 1.0 LOGIK ---
    # Wir nehmen Rank 49 (Alpha=1, Beta=0) und duplizieren ihn für alle Beta-Werte
    print("Applying Alpha=1.0 mathematical equivalence logic...")
    a1_data = df[df['alpha'] == 1.0].copy()
    if not a1_data.empty:
        expanded_a1 = []
        for b in BETA_STEPS:
            if b == 0.0: continue # Original Rank 49 ist schon drin
            temp = a1_data.copy()
            temp['beta'] = b
            expanded_a1.append(temp)
        df = pd.concat([df] + expanded_a1, ignore_index=True)

    # Aggregierung
    agg = df.groupby(['alpha', 'beta', 'dir']).agg({'snr':'mean', 'tii':'mean', 'dmu':'mean'}).reset_index()
    jitter_df = df.groupby(['alpha', 'beta', 'dir']).apply(lambda x: np.std(x['mu_g'])/np.std(x['mu_p']) if np.std(x['mu_p'])>0 else 1.0).reset_index(name='jitter')
    return pd.merge(agg, jitter_df, on=['alpha', 'beta', 'dir'])

# =====================================================
# 4. PLOTTING (Bleibt gleich)
# =====================================================
def plot_heatmaps(df, subfolder_name):
    if df.empty: return
    target_dir = OUT_DIR / subfolder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {'snr': ('SNR Gain [dB]', 'plasma'), 
               'tii': ('Total Intensity Ratio', 'RdBu_r'), 
               'dmu': ('Positional Shift [px]', 'magma')}
    
    for m_key, (m_title, cmap) in metrics.items():
        for d in ["H", "K", "L"]:
            sub = df[df['dir'] == d]
            if len(sub) < 4: continue 
            
            xi, yi = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            # Griddata interpoliert jetzt sauber bis alpha=1
            zi = griddata((sub['alpha'], sub['beta']), sub[m_key], (xi, yi), method='cubic')
            zi_clean = np.nan_to_num(zi, nan=np.nanmin(zi))

            plt.figure(figsize=(10, 8))
            plt.contourf(xi, yi, zi, levels=50, cmap=cmap)
            plt.colorbar(label=m_title)
            plt.contour(xi, yi, zi, levels=15, colors='white', alpha=0.3)
            plt.scatter(sub['alpha'], sub['beta'], c='white', edgecolors='black', s=20)
            plt.title(f"{m_title} - Direction {d} ({subfolder_name})")
            plt.xlabel("Alpha (SSIM)")
            plt.ylabel("Beta (MSE/MAE)")
            plt.savefig(target_dir / f"Heatmap_2D_{m_key}_{d}.png", dpi=300)
            plt.close()

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(zi_clean, cmap=plt.get_cmap(cmap), vert_exag=0.1, blend_mode='soft')
            ax.plot_surface(xi, yi, zi_clean, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
            ax.view_init(elev=25, azim=250)
            ax.set_title(f"3D {m_title} - {d}\n({subfolder_name})")
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Beta')
            plt.savefig(target_dir / f"Topology_3D_{m_key}_{d}.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    print("Collecting all metrics...")
    df_final = collect_all_metrics()
    
    if not df_final.empty:
        print("\n--- Generating plots for ALL runs (with Alpha=1 expansion) ---")
        plot_heatmaps(df_final, "All_Runs_Full_Rect")
        
        print("\n--- Generating plots WITHOUT Alpha=0.25 runs ---")
        df_filtered = df_final[df_final['alpha'] != 0.25]
        plot_heatmaps(df_filtered, "Filtered_No_Alpha_025")
        
        print(f"\nWorkflow complete. Results in: {OUT_DIR}")
    else:
        print("No metrics collected. Check input files.")