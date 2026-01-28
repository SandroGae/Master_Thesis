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

# Deine 10 Ziel-Paare
VIS_PAIRS = [(5, 15), (11, 20), (12, 18), (13, 1), (15, 19), (16, 17), (21, 19), (22, 17), (29, 25), (50, 13)]

def vis_scaling(image, p_low, p_high):
    v_l, v_h = np.percentile(image, [p_low, p_high])
    return np.clip((image - v_l) / (v_h - v_l + 1e-12), 0, 1)

def run_tuner_for_image(s_idx, img_idx):
    global_idx = (s_idx - 1) * SERIES_LEN + (img_idx - 1)
    with h5py.File(H5_TEST_PATH, "r") as f:
        img_raw = f["high_count/data"][:, :, global_idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.35, left=0.1, right=0.72) 

    img_disp = vis_scaling(img_raw, 0.5, 99.5)
    im = ax.imshow(img_disp, cmap="gray_r", origin="upper")
    ax.set_title(f"ROI TUNER | Serie {s_idx} | Bild {img_idx}", fontsize=14, fontweight='bold')

    # Initial-Patches
    roi_p = patches.Rectangle((60, 0), 21, H, lw=2, ec='blue', fc='none', zorder=10)
    bg_p  = patches.Rectangle((0, 0), 20, H, lw=1, ec='red', fc='red', alpha=0.15)
    fit_p = patches.Rectangle((60, 90), 21, 40, lw=0, fc='green', alpha=0.25)
    ax.add_patch(roi_p); ax.add_patch(bg_p); ax.add_patch(fit_p)

    # --- Slider UI mit exakten Dictionary-Namen ---
    ax_x_start = plt.axes([0.15, 0.26, 0.40, 0.025])
    ax_x_width = plt.axes([0.15, 0.22, 0.40, 0.025])
    ax_bg_gap  = plt.axes([0.15, 0.18, 0.40, 0.025])
    ax_fit_mid = plt.axes([0.15, 0.14, 0.40, 0.025])
    ax_pmin    = plt.axes([0.15, 0.06, 0.15, 0.02])
    ax_pmax    = plt.axes([0.40, 0.06, 0.15, 0.02])

    # Slider Beschriftungen angepasst an SERIES_CONFIG
    s_x_start = Slider(ax_x_start, 'roi_x start ', 0, W-21, valinit=60, valstep=1)
    s_x_width = Slider(ax_x_width, 'roi_x breite', 1, 150, valinit=21, valstep=1)
    s_bg_gap  = Slider(ax_bg_gap,  'bg_gap      ', -150, 150, valinit=-81, valstep=1)
    s_fit_mid = Slider(ax_fit_mid, 'fit_y mitte ', 20, H-20, valinit=110, valstep=1)
    s_pmin    = Slider(ax_pmin, 'vis_p low', 0.0, 10.0, valinit=0.5)
    s_pmax    = Slider(ax_pmax, 'vis_p high', 80.0, 100.0, valinit=99.5)

    info_box = fig.text(0.74, 0.45, "", fontsize=11, family='monospace', 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', pad=10))

    def update(val):
        # 1. Hard-Limit X-Start basierend auf Breite
        max_x = W - s_x_width.val
        s_x_start.valmax = max_x
        ax_x_start.set_xlim(0, max_x)
        if s_x_start.val > max_x: s_x_start.set_val(max_x)
        
        # 2. Hard-Limit BG-Gap basierend auf ROI-Ende
        roi_end = s_x_start.val + s_x_width.val
        min_g, max_g = -roi_end, W - 20 - roi_end
        s_bg_gap.valmin, s_bg_gap.valmax = min_g, max_g
        ax_bg_gap.set_xlim(min_g, max_g)
        if s_bg_gap.val < min_g: s_bg_gap.set_val(min_g)
        if s_bg_gap.val > max_g: s_bg_gap.set_val(max_g)

        # 3. Visuelle Updates
        im.set_data(vis_scaling(img_raw, s_pmin.val, s_pmax.val))
        roi_p.set_x(s_x_start.val); roi_p.set_width(s_x_width.val)
        bg_p.set_x(roi_end + s_bg_gap.val)
        fit_p.set_x(s_x_start.val); fit_p.set_width(s_x_width.val); fit_p.set_y(s_fit_mid.val - 20)

        # 4. Info Panel Text (Direktes Dictionary Mapping)
        out = (
            f"SERIE: {s_idx}\n"
            f"slice_idx: {img_idx-1}\n"
            f"{'='*22}\n"
            f"roi_x:     ({int(s_x_start.val)}, {int(roi_end)})\n"
            f"roi_y:     (0, 192)\n"
            f"bg_gap:    {int(s_bg_gap.val)}\n"
            f"fit_window:({int(s_fit_mid.val-20)}, {int(s_fit_mid.val+20)})\n"
            f"vis_p:     ({s_pmin.val:.1f}, {s_pmax.val:.1f})\n"
            f"{'='*22}\n"
            f"CHECK:\n"
            f"Breite X:  {int(s_x_width.val)} px\n"
            f"BG X-Start:{int(roi_end + s_bg_gap.val)} px"
        )
        info_box.set_text(out)
        fig.canvas.draw_idle()

    for s in [s_x_start, s_x_width, s_bg_gap, s_fit_mid, s_pmin, s_pmax]:
        s.on_changed(update)

    update(None)
    plt.show()

def main():
    for s_idx, img_idx in VIS_PAIRS:
        run_tuner_for_image(s_idx, img_idx)

if __name__ == "__main__":
    main()