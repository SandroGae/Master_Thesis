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

# Deine fixierten vis_p Werte (analog zur L-Richtung)
FIXED_VIS_CONFIG = {
    5: (0.5, 97.5), 11: (0.5, 97.5), 12: (0.5, 98.0), 15: (0.5, 98.5), 16: (0.5, 97.5),
    21: (0.5, 98.0), 22: (0.5, 98.0), 29: (0.5, 95.0), 35: (0.5, 90.0), 50: (0.5, 96.0),
    1: (0.5, 95.0), 13: (0.5, 88.0), 17: (0.5, 95.0), 30: (0.5, 97.0), 32: (0.5, 92.0),
    36: (0.5, 95.0), 38: (0.5, 90.0), 41: (0.5, 94.5), 42: (0.5, 90.0), 45: (0.5, 97.0),
    46: (0.5, 90.0), 47: (0.5, 95.5), 51: (0.5, 94.0), 53: (0.5, 95.0), 55: (0.5, 95.0),
    56: (0.5, 95.0), 57: (0.5, 98.0), 59: (0.5, 96.0), 64: (0.5, 95.0), 67: (0.5, 95.0),
    68: (0.5, 95.0), 71: (0.5, 95.0), 72: (0.5, 94.0), 73: (0.5, 95.0), 74: (0.5, 96.0)
}

VIS_PAIRS = [
    # (1, 12), (5, 15), (11, 20), (12, 18), (13, 8), (15, 19), (16, 17), (17, 18), 
    # (21, 19), (22, 17), (29, 25), (30, 15), (32, 32), (35, 24), (36, 16), (38, 36), 
    (41, 19), (42, 36), (45, 21), (46, 38), (47, 4), (50, 13), (51, 23), (53, 11), 
    (55, 22), (56, 10), (57, 23), (59, 16), (64, 6), (67, 11), (68, 21), (71, 27), 
    (72, 16), (73, 26), (74, 24)
]

def vis_scaling(image, p_low, p_high):
    v_l, v_h = np.percentile(image, [p_low, p_high])
    return np.clip((image - v_l) / (v_h - v_l + 1e-12), 0, 1)

def run_tuner_h_direction(s_idx, img_idx):
    # Hardcoded vis_p für diese Serie abrufen
    p_low, p_high = FIXED_VIS_CONFIG.get(s_idx, (0.5, 98.0))

    # Korrektur des globalen Index (img_idx - 1)
    global_idx = (s_idx - 1) * SERIES_LEN + (img_idx - 1)
    with h5py.File(H5_TEST_PATH, "r") as f:
        img_raw = f["high_count/data"][:, :, global_idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.20, left=0.1, right=0.70) 

    img_disp = vis_scaling(img_raw, p_low, p_high)
    im = ax.imshow(img_disp, cmap="gray_r", origin="upper")
    ax.set_title(f"H-DIRECTION TUNER | Serie {s_idx} | Bild {img_idx} (vis_p fixiert)", fontsize=14, fontweight='bold')

    # Patches initialisieren
    roi_p = patches.Rectangle((0, 0), ROI_W, ROI_H, lw=2, ec='blue', fc='none', zorder=10)
    bg_top_p = patches.Rectangle((0, 0), ROI_W, 10, lw=1, ec='red', fc='red', alpha=0.2)
    bg_bot_p = patches.Rectangle((0, 0), ROI_W, 10, lw=1, ec='red', fc='red', alpha=0.2)
    
    ax.add_patch(roi_p); ax.add_patch(bg_top_p); ax.add_patch(bg_bot_p)

    # --- Slider UI (vis_p entfernt!) ---
    ax_x_pos  = plt.axes([0.15, 0.12, 0.35, 0.025])
    ax_y_pos  = plt.axes([0.15, 0.08, 0.35, 0.025])
    ax_bg_gap = plt.axes([0.60, 0.12, 0.20, 0.025])
    ax_bg_h   = plt.axes([0.60, 0.08, 0.20, 0.025])
    
    s_x_pos  = Slider(ax_x_pos,  'X-Position ', 0, W - ROI_W, valinit=60, valstep=1)
    s_y_pos  = Slider(ax_y_pos,  'Y-Position ', 0, H - ROI_H, valinit=102, valstep=1)
    s_bg_gap = Slider(ax_bg_gap, 'Lücke (gap)', 0, 50, valinit=5, valstep=1)
    s_bg_h   = Slider(ax_bg_h,   'BG Höhe    ', 1, 50, valinit=10, valstep=1)

    info_box = fig.text(0.72, 0.40, "", fontsize=10, family='monospace', 
                        va='bottom', ha='left',
                        bbox=dict(facecolor='#fdfdfd', alpha=1.0, edgecolor='#333333', 
                        boxstyle='round,pad=1'))

    def update(val):
        x, y = int(s_x_pos.val), int(s_y_pos.val)
        gap, bgh = int(s_bg_gap.val), int(s_bg_h.val)

        # ROI & Background
        roi_p.set_xy((x, y))
        bg_top_y_bottom = max(0, y - gap)
        bg_top_y_top    = max(0, bg_top_y_bottom - bgh)
        bg_top_p.set_xy((x, bg_top_y_top))
        bg_top_p.set_height(bg_top_y_bottom - bg_top_y_top)

        bg_bot_y_top    = min(H, y + ROI_H + gap)
        bg_bot_y_bottom = min(H, bg_bot_y_top + bgh)
        bg_bot_p.set_xy((x, bg_bot_y_top))
        bg_bot_p.set_height(bg_bot_y_bottom - bg_bot_y_top)

        # Formatierter Output
        out = (
            f"SERIES_CONFIG ENTRY:\n"
            f"{'─'*30}\n"
            f"{s_idx:2d}: {{\n"
            f"    \"slice_idx\": {img_idx-1},\n"
            f"    \"roi_x\":     ({x:3d}, {x + ROI_W:3d}),\n"
            f"    \"roi_y\":     ({y:3d}, {y + ROI_H:3d}),\n"
            f"    \"bg_gap\":    {gap:2d},\n"
            f"    \"bg_h\":      {bgh:2d},\n"
            f"    \"vis_p\":     ({p_low:.1f}, {p_high:.1f}) [FIX],\n"
            f"    \"fit_win\":   (2, 38)\n"
            f"}},\n"
            f"{'─'*30}\n"
            f"H-RICHTUNG | ROI: {ROI_W}x{ROI_H}"
        )
        info_box.set_text(out)
        fig.canvas.draw_idle()

    for s in [s_x_pos, s_y_pos, s_bg_gap, s_bg_h]:
        s.on_changed(update)

    update(None)
    plt.show()

def main():
    print("--- H-DIRECTION ROI TUNER GESTARTET ---")
    for s_idx, img_idx in VIS_PAIRS:
        run_tuner_h_direction(s_idx, img_idx)

if __name__ == "__main__":
    main()