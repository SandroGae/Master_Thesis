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

# Deine fixierten vis_p Werte aus der L-Richtung
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

def run_tuner_for_image(s_idx, img_idx):
    # Hardcoded vis_p für diese Serie abrufen
    p_low, p_high = FIXED_VIS_CONFIG.get(s_idx, (0.5, 99.5))

    global_idx = (s_idx - 1) * SERIES_LEN + (img_idx - 1)
    with h5py.File(H5_TEST_PATH, "r") as f:
        img_raw = f["high_count/data"][:, :, global_idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.72) 

    img_disp = vis_scaling(img_raw, p_low, p_high)
    im = ax.imshow(img_disp, cmap="gray_r", origin="upper")
    ax.set_title(f"ROI TUNER | Serie {s_idx} | Bild {img_idx} (vis_p fixiert: {p_low}/{p_high})", 
                 fontsize=14, fontweight='bold')

    roi_p = patches.Rectangle((60, 0), 21, H, lw=2, ec='blue', fc='none', zorder=10)
    bg_p  = patches.Rectangle((0, 0), 20, H, lw=1, ec='red', fc='red', alpha=0.15)
    fit_p = patches.Rectangle((60, 90), 21, 40, lw=0, fc='green', alpha=0.25)
    ax.add_patch(roi_p); ax.add_patch(bg_p); ax.add_patch(fit_p)

    # Slider UI (vis_p entfernt!)
    ax_x_start = plt.axes([0.15, 0.18, 0.40, 0.025])
    ax_x_width = plt.axes([0.15, 0.14, 0.40, 0.025])
    ax_bg_gap  = plt.axes([0.15, 0.10, 0.40, 0.025])
    ax_fit_mid = plt.axes([0.15, 0.06, 0.40, 0.025])

    s_x_start = Slider(ax_x_start, 'roi_x start ', 0, W-21, valinit=60, valstep=1)
    s_x_width = Slider(ax_x_width, 'roi_x breite', 1, 150, valinit=21, valstep=1)
    s_bg_gap  = Slider(ax_bg_gap,  'bg_gap      ', -150, 150, valinit=-81, valstep=1)
    s_fit_mid = Slider(ax_fit_mid, 'fit_y mitte ', 20, H-20, valinit=110, valstep=1)

    info_box = fig.text(0.74, 0.45, "", fontsize=11, family='monospace', 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue', pad=10))

    def update(val):
        max_x = W - s_x_width.val
        s_x_start.valmax = max_x
        ax_x_start.set_xlim(0, max_x)
        if s_x_start.val > max_x: s_x_start.set_val(max_x)
        
        roi_end = s_x_start.val + s_x_width.val
        min_g, max_g = -roi_end, W - 20 - roi_end
        s_bg_gap.valmin, s_bg_gap.valmax = min_g, max_g
        ax_bg_gap.set_xlim(min_g, max_g)
        if s_bg_gap.val < min_g: s_bg_gap.set_val(min_g)
        if s_bg_gap.val > max_g: s_bg_gap.set_val(max_g)

        # Helligkeit bleibt fix!
        roi_p.set_x(s_x_start.val); roi_p.set_width(s_x_width.val)
        bg_p.set_x(roi_end + s_bg_gap.val)
        fit_p.set_x(s_x_start.val); fit_p.set_width(s_x_width.val); fit_p.set_y(s_fit_mid.val - 20)

        out = (
            f"SERIE: {s_idx}\n"
            f"slice_idx: {img_idx-1}\n"
            f"{'='*22}\n"
            f"roi_x:     ({int(s_x_start.val)}, {int(roi_end)})\n"
            f"roi_y:     (0, 192)\n"
            f"bg_gap:    {int(s_bg_gap.val)}\n"
            f"fit_window:({int(s_fit_mid.val-20)}, {int(s_fit_mid.val+20)})\n"
            f"vis_p:     ({p_low:.1f}, {p_high:.1f}) [FIX]\n"
            f"{'='*22}\n"
            f"CHECK:\n"
            f"Breite X:  {int(s_x_width.val)} px\n"
            f"BG X-Start:{int(roi_end + s_bg_gap.val)} px"
        )
        info_box.set_text(out)
        fig.canvas.draw_idle()

    for s in [s_x_start, s_x_width, s_bg_gap, s_fit_mid]:
        s.on_changed(update)

    update(None)
    plt.show()

def main():
    for s_idx, img_idx in VIS_PAIRS:
        run_tuner_for_image(s_idx, img_idx)

if __name__ == "__main__":
    main()