#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Konfiguration
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Paper_Recreation"

NPZ_FILE = "Pred_unet_25d_SSIM_middle_improved_V2_D5_S12_FullSeries.npz"
PLOT_MIN_LIMIT = -0.5  # Clip
FRAME_START, FRAME_END = 10, 21 # Frame-Bereich (Slice Auswahl)


# ROI
CFG_H = {
    'ROI_X': (65, 86), 'ROI_Y': (104, 115),
    'BG_GAP': 5, 'BG_H': 10, 'TYPE': 'box_vertical_bg'
}
CFG_K = {
    'ROI_X': (60, 81), 'ROI_Y': (0, 192), 
    'BG_GAP': 80, 'BG_W': 10, 'IMG_W': 240, 'TYPE': 'strip_horizontal_bg'
}
CFG_L = {
    'ROI_X': (0, 240), 'ROI_Y': (102, 117),
    'BG_GAP': 5, 'BG_H': 10, 'TYPE': 'strip_vertical_bg'
}

# Berechnung
def calculate_srbr(image, config):
    x1, x2 = config['ROI_X']
    y1, y2 = config['ROI_Y']
    
    # Signal
    signal_roi = image[y1:y2, x1:x2]
    sum_signal = np.sum(signal_roi)
    n_signal = signal_roi.size
    
    # Background
    bg_pixels = []
    if config['TYPE'] in ['box_vertical_bg', 'strip_vertical_bg']:
        gap, h_bg = config['BG_GAP'], config['BG_H']
        yt1, yt2 = max(0, y1 - gap - h_bg), max(0, y1 - gap)
        if yt2 > yt1: bg_pixels.append(image[yt1:yt2, x1:x2])
        h_img = image.shape[0]
        yb1, yb2 = min(h_img, y2 + gap), min(h_img, y2 + gap + h_bg)
        if yb2 > yb1: bg_pixels.append(image[yb1:yb2, x1:x2])

    elif config['TYPE'] == 'strip_horizontal_bg':
        gap, w_bg, w_img = config['BG_GAP'], config['BG_W'], config.get('IMG_W', 240)
        xr1 = min(w_img, x2 + gap)
        xr2 = min(w_img, xr1 + w_bg)
        if xr2 > xr1: bg_pixels.append(image[y1:y2, xr1:xr2])
            
    if not bg_pixels: return None
        
    bg_concat = np.concatenate(bg_pixels)
    mean_bg = np.mean(bg_concat)
    if mean_bg <= 1e-12: mean_bg = 1e-12
    
    sum_bg_equivalent = mean_bg * n_signal
    srbr = (sum_signal - sum_bg_equivalent) / sum_bg_equivalent
    return srbr

def analyze_series(vol_lc, vol_pred, vol_gt, config):
    res_gt, res_lc, res_pred = [], [], []
    n_frames = vol_lc.shape[0]
    
    for i in range(n_frames):
        val_gt   = calculate_srbr(vol_gt[i], config)
        val_lc   = calculate_srbr(vol_lc[i], config)
        val_pred = calculate_srbr(vol_pred[i], config)
        
        if val_gt is not None: 
            res_gt.append(val_gt)
            res_lc.append(val_lc)
            res_pred.append(val_pred)
            
    return res_gt, res_lc, res_pred


def main():
    file_path = IN_DIR / NPZ_FILE
    if not file_path.exists():
        file_path = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw" / NPZ_FILE
    
    print(f"Lade: {file_path}")
    data = np.load(file_path)
    
    # Rohdaten laden
    raw_lc = data['lc'].astype(np.float32)
    raw_pred = data['pred'].astype(np.float32)
    raw_gt = data['gt'].astype(np.float32)
    
    total_frames = raw_lc.shape[0]
    
    # slicing: Gewünschten Bereich ausschneiden
    start = max(0, FRAME_START)
    end   = min(total_frames, FRAME_END + 1) # +1 für inklusiv
    
    vol_lc   = raw_lc[start:end]
    vol_pred = raw_pred[start:end]
    vol_gt   = raw_gt[start:end]
    
    # Analyse starten (nur auf dem Slice)
    h_data = analyze_series(vol_lc, vol_pred, vol_gt, CFG_H)
    k_data = analyze_series(vol_lc, vol_pred, vol_gt, CFG_K)
    l_data = analyze_series(vol_lc, vol_pred, vol_gt, CFG_L)
    
    # PLOTTING
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)
    
    configs = [
        ('h', h_data, 'Analysis along $h$ (Box)'),
        ('k', k_data, 'Analysis along $k$ (Strip)'),
        ('l', l_data, 'Analysis along $\ell$ (Strip)')
    ]
    
    for ax, (name, (x_gt, y_lc, y_pred), title) in zip(axes, configs):
        
        # Filtern (Alles unter -0.5 weg)
        filt_gt, filt_lc, filt_pred = [], [], []
        for g, l, p in zip(x_gt, y_lc, y_pred):
            if g > PLOT_MIN_LIMIT and l > PLOT_MIN_LIMIT and p > PLOT_MIN_LIMIT:
                filt_gt.append(g)
                filt_lc.append(l)
                filt_pred.append(p)
        
        if not filt_gt:
            ax.set_title(f"{title}\n(Keine Daten > {PLOT_MIN_LIMIT})")
            continue

        # Automatische Skalierung (Zoom auf Inhalt)
        all_vals = filt_gt + filt_lc + filt_pred
        real_min = min(all_vals)
        real_max = max(all_vals)
        
        # 10% Puffer hinzufügen
        span = real_max - real_min
        if span == 0: span = 0.1
        
        limit_min = real_min - span * 0.1
        limit_max = real_max + span * 0.1
        
        # Diagonale
        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.5, label='Ideal')
        
        # Nulllinien
        if limit_min < 0 and limit_max > 0:
            ax.axhline(0, color='gray', lw=0.8, alpha=0.3)
            ax.axvline(0, color='gray', lw=0.8, alpha=0.3)

        # Scatter
        ax.scatter(filt_gt, filt_lc, c='indianred', marker='*', s=80, alpha=0.7, label='Low Count')
        ax.scatter(filt_gt, filt_pred, c='cornflowerblue', marker='o', s=50, alpha=0.8, label='Denoised')
        
        # Labels
        ax.set_xlabel('SRBR (HC / Ground Truth)', fontsize=12, fontweight='bold')
        if name == 'h': ax.set_ylabel('SRBR (Observed)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14)
        ax.grid(True, ls=':', alpha=0.6)
        
        # Achsen setzen
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        
        if name == 'h': ax.legend(loc='upper left', frameon=True)
        ax.text(-0.1, 1.05, name, transform=ax.transAxes, size=20, weight='bold')

    plt.tight_layout()
    out_name = f"Paper_Fig3_Sliced_{start}-{end}_{Path(NPZ_FILE).stem}.png"
    plt.savefig(OUT_DIR / out_name)
    plt.show()
    print(f"Grafik gespeichert: {OUT_DIR / out_name}")

if __name__ == "__main__":
    main()

