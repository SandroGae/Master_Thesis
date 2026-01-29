#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from pathlib import Path

# =====================================================
# Konfiguration & Pfade
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
W, H = 240, 192 
SERIES_LEN = 41
BG_BOX_H = 10 # Höhe der Hintergrund-Streifen

# Deine 10 Ziel-Paare
VIS_PAIRS = [(35, 24), (5, 15), (11, 20), (12, 18), (15, 19), (16, 17), (21, 19), (22, 17), (29, 25), (50, 13)]

def vis_scaling(image, p_low, p_high):
    v_l, v_h = np.percentile(image, [p_low, p_high])
    return np.clip((image - v_l) / (v_h - v_l + 1e-12), 0, 1)

def run_tuner_for_image_L(s_idx, img_idx):
    global_idx = (s_idx - 1) * SERIES_LEN + (img_idx - 1)
    with h5py.File(H5_TEST_PATH, "r") as f:
        img_raw = f["high_count/data"][:, :, global_idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.35, left=0.1, right=0.72) 

    img_disp = vis_scaling(img_raw, 0.5, 99.5)
    im = ax.imshow(img_disp, cmap="gray_r", origin="upper")
    ax.set_title(f"L-DIRECTION TUNER | Serie {s_idx} | Bild {img_idx}", fontsize=14, fontweight='bold')

    # Initial-Patches (Horizontal orientiert)
    # ROI: Ein horizontaler Streifen über die volle Breite
    roi_p = patches.Rectangle((0, 102), W, 15, lw=2, ec='blue', fc='none', zorder=10)
    # BG: Zwei Streifen (Oben und Unten)
    bg1_p = patches.Rectangle((0, 87), W, BG_BOX_H, lw=1, ec='red', fc='red', alpha=0.15)
    bg2_p = patches.Rectangle((0, 122), W, BG_BOX_H, lw=1, ec='red', fc='red', alpha=0.15)
    # Fit: Vertikale Box markiert den horizontalen Fit-Bereich
    fit_p = patches.Rectangle((20, 102), 100, 15, lw=0, fc='green', alpha=0.25)
    
    ax.add_patch(roi_p); ax.add_patch(bg1_p); ax.add_patch(bg2_p); ax.add_patch(fit_p)

    # --- Slider UI ---
    ax_y_start = plt.axes([0.15, 0.26, 0.40, 0.025])
    ax_y_height = plt.axes([0.15, 0.22, 0.40, 0.025])
    ax_bg_gap   = plt.axes([0.15, 0.18, 0.40, 0.025])
    ax_fit_mid  = plt.axes([0.15, 0.14, 0.40, 0.025])
    ax_pmin     = plt.axes([0.15, 0.06, 0.15, 0.02])
    ax_pmax     = plt.axes([0.40, 0.06, 0.15, 0.02])

    # Slider Beschriftungen für L-Richtung
    s_y_start  = Slider(ax_y_start,  'roi_y start ', 0, H-15, valinit=102, valstep=1)
    s_y_height = Slider(ax_y_height, 'roi_y höhe  ', 1, 50, valinit=15, valstep=1)
    s_bg_gap   = Slider(ax_bg_gap,   'bg_gap      ', 0, 50, valinit=5, valstep=1)
    s_fit_mid  = Slider(ax_fit_mid,  'fit_x mitte ', 50, W-50, valinit=70, valstep=1)
    s_pmin     = Slider(ax_pmin, 'vis_p low', 0.0, 10.0, valinit=0.5)
    s_pmax     = Slider(ax_pmax, 'vis_p high', 80.0, 100.0, valinit=99.5)

    info_box = fig.text(0.74, 0.45, "", fontsize=11, family='monospace', 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', pad=10))

    def update(val):
        # 1. Hard-Limit Y-Start basierend auf Höhe
        max_y = H - s_y_height.val
        s_y_start.valmax = max_y
        ax_y_start.set_xlim(0, max_y)
        if s_y_start.val > max_y: s_y_start.set_val(max_y)
        
        y_s = s_y_start.val
        y_h = s_y_height.val
        y_end = y_s + y_h
        
        # 2. Hard-Limit BG-Gap (Sandwich muss im Bild [0, 192] bleiben)
        # Oben: y_s - gap - BG_BOX_H >= 0  => gap <= y_s - BG_BOX_H
        # Unten: y_end + gap + BG_BOX_H <= 192 => gap <= 192 - y_end - BG_BOX_H
        max_g = min(max(0, y_s - BG_BOX_H), max(0, H - y_end - BG_BOX_H))
        s_bg_gap.valmax = max_g
        ax_bg_gap.set_xlim(0, max_g)
        if s_bg_gap.val > max_g: s_bg_gap.set_val(max_g)

        gap = s_bg_gap.val
        f_mid = s_fit_mid.val
        # Fit Window ist 100px breit (analog zu FIT_WINDOW_L = (20, 120))
        f_start = f_mid - 50
        f_end = f_mid + 50

        # 3. Visuelle Updates
        im.set_data(vis_scaling(img_raw, s_pmin.val, s_pmax.val))
        
        # ROI Update (X ist hier voll 0-240)
        roi_p.set_y(y_s); roi_p.set_height(y_h)
        
        # BG Streifen (Oben & Unten)
        bg1_p.set_y(y_s - gap - BG_BOX_H)
        bg2_p.set_y(y_end + gap)
        
        # Fit Bereich (Grüne Box zeigt horizontalen Fit-Bereich an)
        fit_p.set_x(f_start); fit_p.set_y(y_s); fit_p.set_height(y_h)

        # 4. Info Panel Text
        out = (
            f"SERIE: {s_idx}\n"
            f"slice_idx: {img_idx-1}\n"
            f"{'='*22}\n"
            f"roi_x:     (0, 240)\n"
            f"roi_y:     ({int(y_s)}, {int(y_end)})\n"
            f"bg_gap:    {int(gap)}\n"
            f"fit_window:({int(f_start)}, {int(f_end)})\n"
            f"vis_p:     ({s_pmin.val:.1f}, {s_pmax.val:.1f})\n"
            f"{'='*22}\n"
            f"CHECK:\n"
            f"ROI Höhe:  {int(y_h)} px\n"
            f"BG Oben Y: {int(y_s - gap - BG_BOX_H)}\n"
            f"BG Unten Y:{int(y_end + gap + BG_BOX_H)}"
        )
        info_box.set_text(out)
        fig.canvas.draw_idle()

    for s in [s_y_start, s_y_height, s_bg_gap, s_fit_mid, s_pmin, s_pmax]:
        s.on_changed(update)

    update(None)
    plt.show()

def main():
    print("--- L-DIRECTION ROI TUNER GESTARTET ---")
    for s_idx, img_idx in VIS_PAIRS:
        run_tuner_for_image_L(s_idx, img_idx)

if __name__ == "__main__":
    main()