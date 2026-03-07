#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_DIR = SCRIPT_DIR.parent / "npz_files"
OUT_DIR = SCRIPT_DIR.parent / "Plots_Full_Unverzerrt"
OUT_DIR.mkdir(exist_ok=True)

# Volle Master-Konfiguration für die Visualisierung (30 Serien)
SERIES_CONFIG = {
    # Block 1
    5:  {"slice_idx": 14, "roi_x": (186, 207), "roi_y": (0, 192), "bg_gap": -176, "vis_p": (0.5, 97.5)},
    11: {"slice_idx": 19, "roi_x": (81, 102),  "roi_y": (0, 192), "bg_gap": 51,   "vis_p": (0.5, 97.5)},
    12: {"slice_idx": 17, "roi_x": (58, 79),   "roi_y": (0, 192), "bg_gap": 73,   "vis_p": (0.5, 98.0)},
    13: {"slice_idx": 7,  "roi_x": (120, 141), "roi_y": (0, 192), "bg_gap": 77,   "vis_p": (0.5, 88.0)},
    15: {"slice_idx": 18, "roi_x": (132, 153), "roi_y": (0, 192), "bg_gap": 67,   "vis_p": (0.5, 98.5)},
    16: {"slice_idx": 16, "roi_x": (118, 139), "roi_y": (0, 192), "bg_gap": 81,   "vis_p": (0.5, 97.5)},
    17: {"slice_idx": 17, "roi_x": (186, 207), "roi_y": (0, 192), "bg_gap": -70,  "vis_p": (0.5, 95.0)},
    22: {"slice_idx": 16, "roi_x": (170, 191), "roi_y": (0, 192), "bg_gap": -177, "vis_p": (0.5, 98.0)},

    # Block 2
    29: {"slice_idx": 24, "roi_x": (57, 78),   "roi_y": (0, 192), "bg_gap": 44,   "vis_p": (0.5, 95.0)},
    30: {"slice_idx": 14, "roi_x": (219, 240), "roi_y": (0, 192), "bg_gap": -171, "vis_p": (0.5, 97.0)},
    32: {"slice_idx": 31, "roi_x": (53, 74),   "roi_y": (0, 192), "bg_gap": -73,  "vis_p": (0.5, 92.0)},
    35: {"slice_idx": 23, "roi_x": (106, 127), "roi_y": (0, 192), "bg_gap": 38,   "vis_p": (0.5, 90.0)},
    36: {"slice_idx": 15, "roi_x": (55, 76),   "roi_y": (0, 192), "bg_gap": 22,   "vis_p": (0.5, 95.0)},
    38: {"slice_idx": 35, "roi_x": (109, 130), "roi_y": (0, 192), "bg_gap": 80,   "vis_p": (0.5, 90.0)},
    41: {"slice_idx": 18, "roi_x": (33, 54),   "roi_y": (0, 192), "bg_gap": 37,   "vis_p": (0.5, 94.5)},
    42: {"slice_idx": 35, "roi_x": (157, 178), "roi_y": (0, 192), "bg_gap": 42,   "vis_p": (0.5, 90.0)},
    45: {"slice_idx": 20, "roi_x": (74, 95),   "roi_y": (0, 192), "bg_gap": 55,   "vis_p": (0.5, 97.0)},
    46: {"slice_idx": 37, "roi_x": (213, 234), "roi_y": (0, 192), "bg_gap": -83,  "vis_p": (0.5, 90.0)},

    # Block 3
    50: {"slice_idx": 12, "roi_x": (96, 117),  "roi_y": (0, 192), "bg_gap": 40,   "vis_p": (0.5, 96.0)},
    51: {"slice_idx": 22, "roi_x": (137, 158), "roi_y": (0, 192), "bg_gap": 56,   "vis_p": (0.5, 94.0)},
    55: {"slice_idx": 20, "roi_x": (177, 198), "roi_y": (0, 192), "bg_gap": -76,  "vis_p": (0.5, 95.0)},
    56: {"slice_idx": 9,  "roi_x": (151, 172), "roi_y": (0, 192), "bg_gap": 48,   "vis_p": (0.5, 95.0)},
    57: {"slice_idx": 22, "roi_x": (180, 201), "roi_y": (0, 192), "bg_gap": -201, "vis_p": (0.5, 98.0)},
    59: {"slice_idx": 15, "roi_x": (190, 211), "roi_y": (0, 192), "bg_gap": -211, "vis_p": (0.5, 96.0)},
    64: {"slice_idx": 5,  "roi_x": (213, 234), "roi_y": (0, 192), "bg_gap": -231, "vis_p": (0.5, 95.0)},
    67: {"slice_idx": 10, "roi_x": (30, 51),   "roi_y": (0, 192), "bg_gap": 48,   "vis_p": (0.5, 95.0)},
    68: {"slice_idx": 20, "roi_x": (27, 48),   "roi_y": (0, 192), "bg_gap": 39,   "vis_p": (0.5, 95.0)},

    # Block 4
    72: {"slice_idx": 15, "roi_x": (43, 64),   "roi_y": (0, 192), "bg_gap": -64,  "vis_p": (0.5, 94.0)},
    73: {"slice_idx": 25, "roi_x": (81, 102),  "roi_y": (0, 192), "bg_gap": 30,   "vis_p": (0.5, 95.0)},
    74: {"slice_idx": 23, "roi_x": (65, 86),   "roi_y": (0, 192), "bg_gap": 113,  "vis_p": (0.5, 96.0)},
}

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    return np.clip((image - vmin) / (max(1e-9, vmax - vmin)), 0, 1)

# =====================================================
# 2. ROBUSTE PLOTTING FUNKTION
# =====================================================
def plot_full_detector_comparison(npz_path):
    try:
        # --- 1. SERIEN-ID EXTRAHIEREN ---
        match = re.search(r"_S(\d+)\.npz", npz_path.name)
        if not match:
            return False
        
        s_id = int(match.group(1))
        
        if s_id not in SERIES_CONFIG:
            print(f"Serie {s_id} übersprungen (Nicht in SERIES_CONFIG)")
            return False

        # --- 2. CONFIG PARAMETER LADEN ---
        cfg = SERIES_CONFIG[s_id]
        slice_z = cfg["slice_idx"]
        p_low, p_high = cfg.get("vis_p", (0.5, 99.5)) # Fallback

        # --- 3. DATEN LADEN & SLICEN ---
        data = np.load(npz_path)
        
        # Inferenz-Skript speichert mu unter 'pred' und sigma unter 'sigma'
        # Shape ist (41, 192, 240)
        full_mu = data['pred'][slice_z]
        full_sigma = data['sigma'][slice_z]

        # --- 4. VISUALISIERUNG ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=200)
        
        # Mu-Plot: Mit individueller vis_norm Skalierung aus dem Dict!
        im0 = axes[0].imshow(vis_norm(full_mu, p_low, p_high), cmap='gray_r', aspect='equal')
        axes[0].set_title(f"Reconstruction ($\\mu$) - Serie {s_id}\nSlice: {slice_z} | Helligkeit: {p_low}-{p_high}%", fontsize=14)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # INTELLIGENTE SIGMA-LOGIK (Perzentil-Clipping)
        # Wir ermitteln das 99. Perzentil (ignoriert die extremsten 1% der Pixel)
        sigma_vmax = np.percentile(full_sigma, 99.0)
        if sigma_vmax == 0: sigma_vmax = 0.01 # Fallback, falls das Bild komplett leer ist

        # Sigma-Plot: Zwingt die Farbskala (vmin, vmax), diese neuen Grenzen zu nutzen
        im1 = axes[1].imshow(full_sigma, cmap='inferno', aspect='equal', vmin=0, vmax=sigma_vmax)
        axes[1].set_title(f"Predictive Uncertainty ($\\sigma$) - Serie {s_id}\nSlice: {slice_z} | Max: {sigma_vmax:.3f}", fontsize=14)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel("Detector X")
            ax.set_ylabel("Detector Y")

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Full_{npz_path.stem}.png", bbox_inches='tight')
        plt.close(fig) 
        return True

    except Exception as e:
        print(f"Fehler bei {npz_path.name}: {e}")
        return False

# =====================================================
# 3. RUNNER
# =====================================================
if __name__ == "__main__":
    npz_files = sorted(list(NPZ_DIR.glob("*.npz")))
    print(f"Gefunden: {len(npz_files)} NPZ-Dateien. Erstelle Full-Frame Plots...")
    
    for f in tqdm(npz_files):
        plot_full_detector_comparison(f)

    print(f"\n>>> Fertig! Die unverzerrten Bilder liegen in: {OUT_DIR}")