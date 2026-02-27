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

# FIXIERTE ROI GRÖSSE (H-Richtung Standard)
ROI_W, ROI_H = 21, 15

VIS_PAIRS = [
    # Bestehende Liste
    (35, 24), (5, 15), (11, 20), (12, 18), (15, 19), 
    (16, 17), (21, 19), (22, 17), (29, 25), (50, 13),
    
    # Ergänzungen aus dem Bild (Obere Reihe)
    (1, 12), (13, 8), (17, 18), (30, 15), (32, 32), 
    (36, 15), (38, 36), (41, 19), (42, 36), (45, 21),
    
    # Ergänzungen aus dem Bild (Mittlere Reihe)
    (47, 6), (51, 23), (53, 11), (55, 21), 
    (56, 10), (57, 23), (59, 16), (64, 6), (67, 11), (68, 21), (71, 27),
    
    # Ergänzungen aus dem Bild (Untere Reihe)
    (72, 16), (73, 26), (74, 24)
]

def vis_scaling(image, p_low, p_high):
    v_l, v_h = np.percentile(image, [p_low, p_high])
    return np.clip((image - v_l) / (v_h - v_l + 1e-12), 0, 1)

def run_tuner_h_direction(s_idx, img_idx):
    global_idx = (s_idx - 1) * SERIES_LEN + img_idx
    with h5py.File(H5_TEST_PATH, "r") as f:
        img_raw = f["high_count/data"][:, :, global_idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.70) 

    img_disp = vis_scaling(img_raw, 0.5, 99.5)
    im = ax.imshow(img_disp, cmap="gray_r", origin="upper")
    ax.set_title(f"H-DIRECTION TUNER (FIXED ROI) | Serie {s_idx} | Slice {img_idx}", fontsize=14, fontweight='bold')

    # Patches initialisieren
    roi_p = patches.Rectangle((0, 0), ROI_W, ROI_H, lw=2, ec='blue', fc='none', zorder=10)
    bg_top_p = patches.Rectangle((0, 0), ROI_W, 10, lw=1, ec='red', fc='red', alpha=0.2)
    bg_bot_p = patches.Rectangle((0, 0), ROI_W, 10, lw=1, ec='red', fc='red', alpha=0.2)
    
    ax.add_patch(roi_p)
    ax.add_patch(bg_top_p)
    ax.add_patch(bg_bot_p)

    # --- Aufgeräumte Slider UI ---
    ax_x_pos  = plt.axes([0.15, 0.15, 0.35, 0.025])
    ax_y_pos  = plt.axes([0.15, 0.11, 0.35, 0.025])
    
    ax_bg_gap = plt.axes([0.60, 0.15, 0.25, 0.025])
    ax_bg_h   = plt.axes([0.60, 0.11, 0.25, 0.025])
    
    ax_pmin   = plt.axes([0.15, 0.04, 0.15, 0.02])
    ax_pmax   = plt.axes([0.35, 0.04, 0.15, 0.02])

    # Slider Definitionen
    s_x_pos  = Slider(ax_x_pos,  'X-Position ', 0, W - ROI_W, valinit=60, valstep=1)
    s_y_pos  = Slider(ax_y_pos,  'Y-Position ', 0, H - ROI_H, valinit=102, valstep=1)
    s_bg_gap = Slider(ax_bg_gap, 'Lücke (gap)', 0, 50, valinit=5, valstep=1)
    s_bg_h   = Slider(ax_bg_h,   'BG Höhe    ', 1, 50, valinit=10, valstep=1)
    s_pmin   = Slider(ax_pmin,   'vis_p low  ', 0.0, 5.0, valinit=0.5)
    s_pmax   = Slider(ax_pmax,   'vis_p high ', 90.0, 100.0, valinit=98.0)

    # Info-Box
    info_box = fig.text(0.72, 0.40, "", fontsize=10, family='monospace', 
                        va='bottom', ha='left',
                        bbox=dict(facecolor='#fdfdfd', alpha=1.0, edgecolor='#333333', 
                        boxstyle='round,pad=1'))

    def update(val):
        x = int(s_x_pos.val)
        y = int(s_y_pos.val)
        gap = int(s_bg_gap.val)
        bgh = int(s_bg_h.val)

        # Bild Helligkeit
        im.set_data(vis_scaling(img_raw, s_pmin.val, s_pmax.val))

        # ROI verschieben
        roi_p.set_xy((x, y))

        # Top Background Box (Positioniert relativ zur ROI)
        bg_top_y_bottom = max(0, y - gap)
        bg_top_y_top    = max(0, bg_top_y_bottom - bgh)
        bg_top_p.set_xy((x, bg_top_y_top))
        bg_top_p.set_height(bg_top_y_bottom - bg_top_y_top)

        # Bottom Background Box (Positioniert relativ zur ROI)
        bg_bot_y_top    = min(H, y + ROI_H + gap)
        bg_bot_y_bottom = min(H, bg_bot_y_top + bgh)
        bg_bot_p.set_xy((x, bg_bot_y_top))
        bg_bot_p.set_height(bg_bot_y_bottom - bg_bot_y_top)

        # Formatierter Output für SERIES_CONFIG
        out = (
            f"SERIES_CONFIG ENTRY:\n"
            f"{'─'*30}\n"
            f"{s_idx:2d}: {{\n"
            f"    \"slice_idx\": {img_idx},\n"
            f"    \"roi_x\":     ({x:3d}, {x + ROI_W:3d}),\n"
            f"    \"roi_y\":     ({y:3d}, {y + ROI_H:3d}),\n"
            f"    \"bg_gap\":    {gap:2d},\n"
            f"    \"bg_h\":      {bgh:2d},\n"
            f"    \"vis_p\":     ({s_pmin.val:4.1f}, {s_pmax.val:4.1f}),\n"
            f"    \"fit_win\":   (2, 38)\n"
            f"}},\n"
            f"{'─'*30}\n"
            f"FIXED ROI: {ROI_W}x{ROI_H} px | BG: {bgh} px"
        )
        info_box.set_text(out)
        fig.canvas.draw_idle()

    for s in [s_x_pos, s_y_pos, s_bg_gap, s_bg_h, s_pmin, s_pmax]:
        s.on_changed(update)

    update(None)
    plt.show()

def main():
    for s_idx, img_idx in VIS_PAIRS:
        run_tuner_h_direction(s_idx, img_idx)

if __name__ == "__main__":
    main()