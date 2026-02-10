import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from pathlib import Path

# =====================================================
# 1. SETUP & CONFIG
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
# Pfad basierend auf deinem Bild
IN_DIR = ROOT_DIR / "Unet/Analysis_ROI/Prediction.npz/Predictions_Raw_new_RERUN"
OUT_FILE = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/Comparison_Params_21_vs_33.mp4"

# Suche rein nach Parametern (a, b, seed)
PARAM_21 = "a0.6667_b0.5000_seed42" 
PARAM_33 = "a0.0000_b0.0000_seed43"

SERIES_CONFIG = {
    5:  {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 7.0), "vis_p": (0.5, 99.0)},
    11: {"roi_x": (0, 240), "roi_y": (100, 119), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (3.0, 8.0), "vis_p": (0.5, 98.0)},
    12: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 4.5), "vis_p": (0.5, 99.0)},
    15: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 5.0), "vis_p": (0.5, 99.0)},
    16: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 7.0), "vis_p": (0.5, 98.0)},
    21: {"roi_x": (0, 240), "roi_y": (101, 118), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (3.0, 5.5), "vis_p": (0.5, 99.5)},
    22: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 6.5), "vis_p": (0.5, 98.5)},
    29: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 5.5), "vis_p": (0.5, 99.0)},
    35: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "y_lim_raw": (2.5, 5.0), "vis_p": (0.5, 98.0)},
    50: {"roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 10, "bg_h": 10, "y_lim_raw": (2.5, 5.0), "vis_p": (0.5, 98.5)},
}

SERIES_IDS = list(SERIES_CONFIG.keys())
TOTAL_FRAMES = len(SERIES_IDS) * 41

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def vis_norm(image, p_low, p_high):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    return np.clip((image - vmin) / (max(1e-6, vmax - vmin)), 0, 1)

# =====================================================
# 3. ANIMATIONS-LOGIK
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white', 
                         gridspec_kw={'height_ratios': [1, 0.6]})

def update(i):
    s_idx, f_idx = i // 41, i % 41
    s_id = SERIES_IDS[s_idx]
    cfg = SERIES_CONFIG[s_id]
    
    # Dateinamen-Muster nach deinem Bild
    file_21 = IN_DIR / f"Pred_DeepScan_{PARAM_21}_D5_S{s_id}_FullSeries.npz"
    file_33 = IN_DIR / f"Pred_DeepScan_{PARAM_33}_D5_S{s_id}_FullSeries.npz"
    
    try:
        d21 = np.load(file_21)['pred'][f_idx]
        d33 = np.load(file_33)['pred'][f_idx]
        
        imgs = [d21, d33]
        params = [PARAM_21, PARAM_33]
        titles = [f"Rang 21: {PARAM_21}", f"Rang 33: {PARAM_33}"]
        
        rx, ry, bg_h = cfg["roi_x"], cfg["roi_y"], cfg["bg_h"]
        r1_t, r2_t = max(0, ry[0] - cfg["bg_gap"] - bg_h), min(192, ry[1] + cfg["bg_gap"])
        x_ax = np.arange(rx[0], rx[1])

        for col in range(2):
            img = imgs[col]
            # Oben: Visualisierung
            ax0 = axes[0, col]; ax0.clear()
            ax0.imshow(vis_norm(img, cfg['vis_p'][0], cfg['vis_p'][1]), cmap="gray_r", origin='lower')
            ax0.add_patch(patches.Rectangle((rx[0], ry[0]), rx[1]-rx[0], ry[1]-ry[0], lw=1.5, ec='blue', fc='none'))
            ax0.axis('off')
            ax0.set_title(titles[col], fontweight='bold', fontsize=12)

            # Unten: Profile
            sig_s = img[ry[0]:ry[1], rx[0]:rx[1]]
            bg_s = np.concatenate([img[r1_t:r1_t+bg_h, rx[0]:rx[1]], img[r2_t:r2_t+bg_h, rx[0]:rx[1]]])
            p_sig = np.sum(sig_s, axis=0)
            p_bg = np.sum(bg_s, axis=0) * (sig_s.shape[0]/bg_s.shape[0])
            
            ax1 = axes[1, col]; ax1.clear()
            ax1.plot(x_ax, p_sig, color='blue', alpha=0.7, label='Signal')
            ax1.plot(x_ax, p_bg, color='red', alpha=0.7, label='Background')
            ax1.set_ylim(cfg["y_lim_raw"])
            ax1.grid(True, alpha=0.2)
            if col == 0: ax1.set_ylabel("Counts")
            ax1.legend(loc='upper right', fontsize=8)

        fig.suptitle(f"Serie S{s_id} | Frame {f_idx+1}/41", fontsize=16, fontweight='bold', y=0.98)
    
    except FileNotFoundError as e:
        print(f"Überspringe Frame: {e}")

# =====================================================
# 4. START (3 FPS)
# =====================================================
print(f"Suche Dateien in {IN_DIR}...")
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=333)
ani.save(OUT_FILE, writer='ffmpeg', fps=3, dpi=100)
print(f"Video gespeichert: {OUT_FILE}")