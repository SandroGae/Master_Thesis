import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. ORDNER-DEFINITIONEN & SETUP
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_DIR = SCRIPT_DIR / "npz_series_22"
OUT_DIR = SCRIPT_DIR / "Series_22_Averaged_Plots"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 2. LAYOUT & FORMATIERUNG (DEIN TUNING-BEREICH)
# =====================================================
# SPALTEN-BREITE (Gilt einheitlich für Low Count, Prediction, Ground Truth)
W_COL = 6.0 

# ZEILEN-HÖHEN (Individuell steuerbar für die 3 Ansichten)
H_VIS = 5.0  # Höhe der Bilder (Zeile 1)
H_RAW = 5.0  # Höhe der Raw Counts (Zeile 2)
H_SBR = 5.0  # Höhe der SBR Plots (Zeile 3)

# ABSTÄNDE (Gaps)
GAP_W = -0.4 # Horizontaler Abstand zwischen den 3 Spalten
GAP_H = -0.4  # Vertikaler Abstand zwischen den 3 Zeilen

DPI = 300

# --- SCHRIFTGRÖSSEN (Tuning) ---
FS_TITLE  = 26    # Haupt-Überschriften (z.B. H: Low Count)
FS_AXIS   = 22    # Achsenbeschriftungen an der Seite (Visualization, etc.)
FS_TICKS  = 16    # Zahlen an den Achsen (0, 10, 20...)
FS_LEGEND = 14    # Text innerhalb der Legende (Gauss-Werte)

# =====================================================
# 3. PHYSIKALISCHE KONFIGURATION (UNBERÜHRT)
# =====================================================
S_ID = 22
CFG = {
    "H": {"z": 16, "roi_x": (172, 193), "roi_y": (102, 117), "bg_gap": 5, "win": (2, 38),   "ylim_raw": (60, 85),  "ylim_sbr": (-0.1, 0.55)},
    "K": {"z": 16, "roi_x": (170, 191), "roi_y": (0, 192),   "bg_gap": -177, "win": (90, 130), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.1, 0.8)},
    "L": {"z": 16, "roi_x": (0, 240),   "roi_y": (102, 117), "bg_gap": 5, "win": (140, 240), "ylim_raw": (2.5, 8.0), "ylim_sbr": (-0.1, 0.5)}
}
ALPHA_GREEN = 0.20
ALPHA_RED   = 0.15
ROW_LABELS = ["Low Count", "Prediction", "Ground Truth"]
FIT_COLORS = ['none', 'mediumseagreen', 'darkviolet']

BEST_AVG_POINT = 34
BEST_SINGLE_POINT = 38
BEST_SINGLE_SEED = 51
ALL_POINTS = range(43)
ALL_SEEDS = range(42, 52)

# =====================================================
# 4. MATHEMATIK & HELFER
# =====================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(img, p_low=0.5, p_high=98.5):
    vmin, vmax = np.percentile(img, [p_low, p_high])
    if vmax - vmin == 0: return img
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

def perform_fit(x, y, y_err, win):
    mask = (x >= win[0]) & (x <= win[1])
    xf, yf, sf = x[mask], y[mask], y_err[mask]
    if len(yf) < 5: return None, None
    p0 = [np.max(yf), xf[np.argmax(yf)], 10.0]
    try:
        popt, pcov = curve_fit(gaussian, xf, yf, p0=p0, sigma=sf, absolute_sigma=True, maxfev=5000)
        return popt, np.sqrt(np.diag(pcov))
    except: return None, None

def get_sbr_profile(vol, direction):
    c = CFG[direction]
    rx, ry = c["roi_x"], c["roi_y"]
    n_sig = (rx[1]-rx[0])*(ry[1]-ry[0])
    
    if direction == "H":
        p_sig = np.sum(vol[:, ry[0]:ry[1], rx[0]:rx[1]], axis=(1, 2))
        bg_pix = np.concatenate([vol[:, ry[0]-15:ry[0]-5, rx[0]:rx[1]], vol[:, ry[1]+5:ry[1]+15, rx[0]:rx[1]]], axis=1)
        p_bg = np.mean(bg_pix, axis=(1, 2)) * n_sig
        x_ax = np.arange(vol.shape[0])
    elif direction == "K":
        img = vol[c["z"]]
        p_sig = np.sum(img[ry[0]:ry[1], rx[0]:rx[1]], axis=1)
        bg_l = min(240 - 21, max(0, rx[1] + c["bg_gap"]))
        bg_pix = img[ry[0]:ry[1], bg_l:bg_l+20]
        p_bg = np.sum(bg_pix, axis=1) * ((rx[1]-rx[0])/20)
        x_ax = np.arange(192)
    else: # L
        img = vol[c["z"]]
        p_sig = np.sum(img[ry[0]:ry[1], rx[0]:rx[1]], axis=0)
        bg_pix = np.concatenate([img[ry[0]-15:ry[0]-5, rx[0]:rx[1]], img[ry[1]+5:ry[1]+15, rx[0]:rx[1]]], axis=0)
        p_bg = np.sum(bg_pix, axis=0) * ((ry[1]-ry[0])/20)
        x_ax = np.arange(240)
    
    denom = np.where(p_bg <= 0, 1e-9, p_bg)
    sbr = (p_sig - p_bg) / denom

    px_std = np.std(bg_pix)
    n_y = n_sig / p_sig.size
    n_bg_y = bg_pix.size / p_bg.size
    err_sig = px_std * np.sqrt(n_y)
    err_bg = px_std * np.sqrt(n_bg_y) * (n_y / n_bg_y)
    err_net = np.sqrt(err_sig**2 + err_bg**2)
    rel_net = err_net / np.maximum(np.abs(p_sig - p_bg), 1e-9)
    rel_bg = err_bg / np.maximum(np.abs(p_bg), 1e-9)
    err = np.abs(sbr) * np.sqrt(rel_net**2 + rel_bg**2)
    
    return x_ax, p_sig, p_bg, sbr, err

def get_averaged_vol(points, seeds):
    sum_vol, lc, gt = None, None, None
    count = 0
    for p in points:
        for s in seeds:
            pattern = f"Eval_P{p:02d}_*_seed{s}_*_S{S_ID}.npz"
            files = list(NPZ_DIR.glob(pattern))
            if not files: continue
            try:
                with np.load(files[0]) as data:
                    p_data = data['pred'].astype(np.float32)
                    if sum_vol is None: 
                        sum_vol = p_data
                        lc, gt = np.array(data['lc']), np.array(data['gt'])
                    else: sum_vol += p_data
                count += 1
            except: continue
    if count > 0:
        return sum_vol / count, lc, gt, count
    else:
        return None, None, None, 0

# =====================================================
# 5. MASTER PLOTTING ENGINE (3x3 TRANSPOSED)
# =====================================================
def plot_master_3x3_transposed(vol_pr, vol_lc, vol_gt, file_suffix, label):
    # Struktur: 3 Spalten (LC, Pred, GT) und 3 Zeilen (Vis, Raw, SBR)
    width_ratios = [W_COL, GAP_W, W_COL, GAP_W, W_COL]
    height_ratios = [H_VIS, GAP_H, H_RAW, GAP_H, H_SBR]

    total_w = sum(width_ratios)
    total_h = sum(height_ratios)

    dirs = ["H", "K", "L"]
    vols = [vol_lc, vol_pr, vol_gt]

    for d_name in dirs:
        d_cfg = CFG[d_name]
        
        fig = plt.figure(figsize=(total_w * 1.5, total_h * 1.5), dpi=DPI)
        gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), 
                               width_ratios=width_ratios, 
                               height_ratios=height_ratios)

        for v_idx, vol in enumerate(vols):
            col_idx = v_idx * 2 # 0, 2, 4 wegen Gaps
            x, sig, bg, sbr, err = get_sbr_profile(vol, d_name)
            
            # --- ZEILE 0: VISUALIZATION ---
            ax0 = fig.add_subplot(gs[0, col_idx])
            img_slice = vol[d_cfg["z"]]
            ax0.imshow(vis_norm(img_slice, p_low=0.5, p_high=98.5), cmap='gray_r', origin='upper', aspect='equal')
            
            # Überschrift oben über den Spalten (z.B. "H: Low Count")
            ax0.set_title(f"{d_name}: {ROW_LABELS[v_idx]}", fontsize=FS_TITLE, fontweight='bold', pad=20)
            
            # Y-Achsen-Label nur für die ganz linke Spalte
            if v_idx == 0:
                ax0.set_ylabel("Visualization", fontsize=FS_AXIS, fontweight='bold', labelpad=15)
            
            # ROI Patches
            rx, ry = d_cfg["roi_x"], d_cfg["roi_y"]
            ax0.add_patch(patches.Rectangle((rx[0], ry[0]), rx[1]-rx[0], ry[1]-ry[0], lw=2, ec='blue', fc='none', zorder=10))
            
            if d_name == "H":
                ax0.add_patch(patches.Rectangle((rx[0], ry[0]), rx[1]-rx[0], ry[1]-ry[0], lw=0, fc='green', alpha=ALPHA_GREEN))
                ax0.add_patch(patches.Rectangle((rx[0], ry[0]-15), rx[1]-rx[0], 10, lw=1, ec='red', fc='red', alpha=ALPHA_RED))
                ax0.add_patch(patches.Rectangle((rx[0], ry[1]+5), rx[1]-rx[0], 10, lw=1, ec='red', fc='red', alpha=ALPHA_RED))
            elif d_name == "L":
                w_s, w_e = d_cfg["win"]
                ax0.add_patch(patches.Rectangle((w_s, ry[0]), w_e-w_s, ry[1]-ry[0], lw=0, fc='green', alpha=ALPHA_GREEN))
                ax0.add_patch(patches.Rectangle((rx[0], ry[0]-15), rx[1]-rx[0], 10, lw=1, ec='red', fc='red', alpha=ALPHA_RED))
                ax0.add_patch(patches.Rectangle((rx[0], ry[1]+5), rx[1]-rx[0], 10, lw=1, ec='red', fc='red', alpha=ALPHA_RED))
            else: # K
                w_s, w_e = d_cfg["win"]
                ax0.add_patch(patches.Rectangle((rx[0], w_s), rx[1]-rx[0], w_e-w_s, lw=0, fc='green', alpha=ALPHA_GREEN))
                bg_l = min(240 - 21, max(0, rx[1] + d_cfg["bg_gap"]))
                ax0.add_patch(patches.Rectangle((bg_l, ry[0]), 20, ry[1]-ry[0], lw=1, ec='red', fc='red', alpha=ALPHA_RED))
            ax0.set_xticks([]); ax0.set_yticks([])

            # --- ZEILE 2: RAW COUNTS ---
            ax1 = fig.add_subplot(gs[2, col_idx])
            ax1.plot(x, sig, color='blue', lw=1.5, label='Raw Sum')
            ax1.plot(x, bg, color='red', lw=1.5, label='Background Sum')
            ax1.axvspan(d_cfg["win"][0], d_cfg["win"][1], color='green', alpha=0.1)
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(d_cfg["ylim_raw"])
            ax1.tick_params(axis='both', which='major', labelsize=FS_TICKS)
            ax1.grid(True, alpha=0.2)
            
            # Legende in der mittleren Spalte (Prediction) hinzufügen
            if v_idx == 1:
                ax1.legend(loc='upper right', fontsize=14)
            
            if v_idx == 0:
                ax1.set_ylabel("Raw Counts", fontsize=FS_AXIS, fontweight='bold', labelpad=15)
            
            if v_idx == 0:
                ax1.set_ylabel("Raw Counts", fontsize=FS_AXIS, fontweight='bold', labelpad=15)

            # --- ZEILE 4: SBR / GAUSS FIT ---
            ax2 = fig.add_subplot(gs[4, col_idx])
            
            # WICHTIG: label='SRBR' hinzugefügt!
            ax2.errorbar(x, sbr, yerr=err, fmt='.', color='black', alpha=0.5, markersize=4, label='SRBR')
            ax2.axhline(0, color='gray', ls=':', alpha=0.5)
            
            if v_idx > 0:
                popt, perr = perform_fit(x, sbr, err, d_cfg["win"])
                if popt is not None:
                    p, e = popt, perr
                    l = (f"Gauss (Amp={p[0]:.2f}$\pm${e[0]:.2f}, "
                         f"Peak={p[1]:.1f}$\pm${e[1]:.1f}, "
                         f"$\sigma$={p[2]:.2f})")
                    
                    ax2.plot(x, gaussian(x, *popt), color=FIT_COLORS[v_idx], lw=2.5, ls='--', label=l)
            
            # Legende IMMER aufrufen (nicht nur, wenn es einen Fit gibt)
            handles, labels = ax2.get_legend_handles_labels()
            # Wenn es zwei Einträge gibt, drehen wir sie um, damit "Gauss" über "SRBR" steht
            if len(handles) > 1:
                handles = handles[::-1]
                labels = labels[::-1]
                
            ax2.legend(handles, labels, fontsize=10, loc='upper right')
            
            ax2.set_xlim(d_cfg["win"])
            ax2.set_ylim(d_cfg["ylim_sbr"])
            ax2.tick_params(axis='both', which='major', labelsize=FS_TICKS)
            ax2.grid(True, alpha=0.2)
            
            if v_idx == 0:
                ax2.set_ylabel("SBR Profile", fontsize=FS_AXIS, fontweight='bold', labelpad=15)

        # Plot pro Richtung speichern
        fig.align_ylabels()
        plt.savefig(OUT_DIR / f"S{S_ID}_Final_Pro_{file_suffix}_{d_name}.png", bbox_inches='tight', dpi=DPI)
        plt.close(fig)

if __name__ == "__main__":
    scenarios = [
        ([BEST_SINGLE_POINT], [BEST_SINGLE_SEED], "Single", "Best Single Model"),
        ([BEST_AVG_POINT], list(ALL_SEEDS), "Point", "Point 10-Seed Average"),
        # MAE ONLY
        ([0], list(ALL_SEEDS), "Point00", "Point 00 10-Seed Average"),
        #MSE ONLY
        ([6], list(ALL_SEEDS), "Point06", "Point 06 10-Seed Average"),
        #SSIM ONLY
        ([42], list(ALL_SEEDS), "Point42", "Point 42 10-Seed Average"),
        #BEST AREA
        ([2], list(ALL_SEEDS), "Point2", "Point 2 10-Seed Average"),
        #BEST QUALITY METRICS
        ([14], list(ALL_SEEDS), "Point14", "Point 14 10-Seed Average"),
        #BEST COMPROMISE
        ([23], list(ALL_SEEDS), "Point23", "Point 23 10-Seed Average"),
        (list(ALL_POINTS), list(ALL_SEEDS), "All", "Global Ensemble Average")
    ]
    
    for p, s, suffix, label in scenarios:
        pr, lc, gt, num_models = get_averaged_vol(p, s)
        
        if pr is not None: 
            plot_master_3x3_transposed(pr, lc, gt, suffix, label)
            print(f"   -> Verwendete Modelle für diesen Plot: {num_models}")