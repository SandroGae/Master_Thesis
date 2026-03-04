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
OUT_DIR = SCRIPT_DIR / "Final_Master_Plots"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# 2. LAYOUT & FORMATIERUNG (DEIN TUNING-BEREICH)
# =====================================================
# BREITEN-PARAMETER (Individuell steuerbar)
W_IMG  = 5  # Breite der Bilder links
W_RAW  = 6.5    # NEU: Breite der mittleren Plots (Raw Counts)
W_SBR  = 6.5    # NEU: Breite der rechten Plots (SBR)

# HÖHEN-PARAMETER
# Wenn du hier den Wert senkst, rücken die Bilder vertikal näher zusammen!
H_ROW  = 8    # Verringert, um vertikalen Leerraum bei den Bildern zu killen

# HORIZONTALE ABSTÄNDE (Gaps)
GAP_W1 = 0.02   # Fast kein Abstand zwischen Bild und Plot
GAP_W2 = 0.1    # Puffer zwischen den beiden Plot-Typen

# VERTIKALE ABSTÄNDE
GAP_H_SMALL = 0.00 # Bilder innerhalb eines Blocks kleben fast aneinander
GAP_H_BIG   = 0.3  # Deutlicher Abstand zwischen H, K und L Blöcken

DPI = 300


# --- SCHRIFTGRÖSSEN (Tuning) ---
FS_TITLE  = 28    # Haupt-Überschriften (Visualization, Raw Counts, etc.)
FS_AXIS   = 22    # Achsenbeschriftungen (H: Prediction, etc.)
FS_TICKS  = 16    # Zahlen an den Achsen (0, 10, 20...)
FS_LEGEND = 14    # Text innerhalb der Legende (Gauss-Werte)

# =====================================================
# 3. PHYSIKALISCHE KONFIGURATION (UNBERÜHRT)
# =====================================================
S_ID = 22
CFG = {
    "H": {"z": 16, "roi_x": (172, 193), "roi_y": (102, 117), "bg_gap": 5, "win": (2, 38),   "ylim_raw": (60, 85),  "ylim_sbr": (-0.1, 0.55)},
    "K": {"z": 16, "roi_x": (170, 191), "roi_y": (0, 192),   "bg_gap": -177, "win": (90, 130), "ylim_raw": (3.5, 7.0), "ylim_sbr": (-0.1, 0.6)},
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
    # Rückgabe von 4 Werten statt bisher 3
    if count > 0:
        return sum_vol / count, lc, gt, count
    else:
        return None, None, None, 0

# =====================================================
# 5. MASTER PLOTTING ENGINE (INDIVIDUAL SPACER GRID)
# =====================================================
def plot_master_vertical(vol_pr, vol_lc, vol_gt, file_suffix, label):
    # WICHTIG: Die Liste hat 5 Elemente, also muss GridSpec auch 5 Spalten haben!
    width_ratios = [W_IMG, GAP_W1, W_RAW, GAP_W2, W_SBR]
    
    h_s, h_b = GAP_H_SMALL, GAP_H_BIG
    height_ratios = [
        H_ROW, h_s, H_ROW, h_s, H_ROW, h_b, 
        H_ROW, h_s, H_ROW, h_s, H_ROW, h_b, 
        H_ROW, h_s, H_ROW, h_s, H_ROW
    ]

    total_w = sum(width_ratios)
    total_h = sum(height_ratios)
    
    # Figsize etwas defensiver berechnen, um "Falscher Parameter" durch Übergröße zu vermeiden
    # Breite ca. 30-40 Zoll, Höhe ca. 50-60 Zoll
    fig = plt.figure(figsize=(total_w * 1.5, total_h * 0.8), dpi=DPI)
    
    # HIER WAR DER FEHLER: Die '5' muss exakt mit len(width_ratios) übereinstimmen!
    gs = gridspec.GridSpec(len(height_ratios), 5, 
                           width_ratios=width_ratios, 
                           height_ratios=height_ratios)

    dirs, vols = ["H", "K", "L"], [vol_lc, vol_pr, vol_gt]

    for d_idx, d_name in enumerate(dirs):
        d_cfg = CFG[d_name]
        for v_idx, vol in enumerate(vols):
            grid_row = d_idx * 6 + v_idx * 2
            x, sig, bg, sbr, err = get_sbr_profile(vol, d_name)
            
            # --- SPALTE 0: BILDER (MIT ALLEN PATCHES) ---
            ax0 = fig.add_subplot(gs[grid_row, 0])
            img_slice = vol[d_cfg["z"]]
            ax0.imshow(vis_norm(img_slice, p_low=0.5, p_high=98.5), cmap='gray_r', origin='upper', aspect='equal')
            ax0.set_ylabel(f"{d_name}: {ROW_LABELS[v_idx]}", fontsize=FS_AXIS, fontweight='bold')
            if grid_row == 0: ax0.set_title("Visualization", fontsize=18, fontweight='bold', pad=15)
            
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

            # --- SPALTE 2: RAW COUNTS ---
            ax1 = fig.add_subplot(gs[grid_row, 2])
            ax1.plot(x, sig, color='blue', lw=1.5)
            ax1.plot(x, bg, color='red', lw=1.5)
            ax1.axvspan(d_cfg["win"][0], d_cfg["win"][1], color='green', alpha=0.1) # GRÜNE FLÄCHE HIER
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(d_cfg["ylim_raw"])
            ax1.tick_params(axis='both', which='major', labelsize=FS_TICKS)
            ax1.grid(True, alpha=0.2)

            if grid_row == 0: ax1.set_title("Raw Counts", fontsize=18, fontweight='bold', pad=15)

            # --- SPALTE 4: SBR / GAUSS FIT ---
            ax2 = fig.add_subplot(gs[grid_row, 4])
            ax2.errorbar(x, sbr, yerr=err, fmt='.', color='black', alpha=0.5, markersize=4)
            ax2.axhline(0, color='gray', ls=':', alpha=0.5)
            if v_idx > 0:
                popt, perr = perform_fit(x, sbr, err, d_cfg["win"])
                if popt is not None:
                    l = f"Amplitude: {popt[0]:.2f}\nPeak Position: {popt[1]:.1f}\n$\sigma$: {popt[2]:.2f}"
                    ax2.plot(x, gaussian(x, *popt), color=FIT_COLORS[v_idx], lw=2, ls='--', label=l)
                    ax2.legend(fontsize=FS_LEGEND, loc='upper right')
            ax2.set_xlim(d_cfg["win"])
            ax2.set_ylim(d_cfg["ylim_sbr"])
            ax2.tick_params(axis='both', which='major', labelsize=FS_TICKS)
            ax2.grid(True, alpha=0.2)
            if grid_row == 0: 
                ax0.set_title("Visualization", fontsize=FS_TITLE, fontweight='bold', pad=20)
                ax1.set_title("Raw Counts", fontsize=FS_TITLE, fontweight='bold', pad=20)
                ax2.set_title("SBR Profile", fontsize=FS_TITLE, fontweight='bold', pad=20)

    plt.savefig(OUT_DIR / f"S{S_ID}_Final_Pro_{file_suffix}.png", bbox_inches='tight', dpi=DPI)
    plt.close(fig)

if __name__ == "__main__":
    scenarios = [
        ([BEST_SINGLE_POINT], [BEST_SINGLE_SEED], "Single", "Best Single Model"),
        ([BEST_AVG_POINT], list(ALL_SEEDS), "Point", "Point 10-Seed Average"),
        (list(ALL_POINTS), list(ALL_SEEDS), "All", "Global Ensemble Average")
    ]
    
    for p, s, suffix, label in scenarios:
        # Hier empfangen wir jetzt num_models
        pr, lc, gt, num_models = get_averaged_vol(p, s)
        
        if pr is not None: 
            plot_master_vertical(pr, lc, gt, suffix, label)
            # Dies ist der gewünschte Print unter der "Saved" Meldung
            print(f"   -> Verwendete Modelle für diesen Plot: {num_models}")