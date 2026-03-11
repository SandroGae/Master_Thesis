#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & CONFIGURATION
# =====================================================
EXP_NAME = "CARE_10_SEEDS"
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR = BASE_DIR / "npz_files" / EXP_NAME
OUT_DIR = BASE_DIR / "Thesis_Plots" / EXP_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Deine bewährte Serien-Konfiguration
SERIES_CONFIG = {
    # Block 1
    5:  {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    11: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (40, 140),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    12: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (27, 127),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    13: {"slice_idx": 7,  "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 0, "bg_h": 10, "fit_window": (80, 180),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 88.0)},
    15: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (95, 195),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.5)},
    16: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (81, 181),  "y_lim_raw": (2.5, 7.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    17: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (136, 236), "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},    
    22: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    29: {"slice_idx": 24, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (23, 123),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    30: {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (160, 240),   "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.0)},

    # Block 2
    32: {"slice_idx": 31, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (21, 121),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 92.0)},
    35: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (67, 167),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    36: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (21, 121),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    38: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (64, 164),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    41: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (98, 113),  "bg_gap": 0, "bg_h": 10, "fit_window": (0, 100),   "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.5)},
    42: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    45: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (41, 141),  "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.0)},
    46: {"slice_idx": 37, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    50: {"slice_idx": 12, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (59, 159),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
    51: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (108, 208), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.0)},

    # Block 3
    55: {"slice_idx": 21, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (122, 222), "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    56: {"slice_idx": 9,  "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (106, 206), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    57: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (136, 236), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    59: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (137, 237), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
    64: {"slice_idx": 5,  "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    67: {"slice_idx": 10, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (0, 100),   "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    68: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (5, 105),   "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    72: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (4, 104),   "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.0)},
    73: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (42, 142), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    74: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 0, "bg_h": 10, "fit_window": (32, 132), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
}

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

# =====================================================
# 3. GENERIERUNG DES THESIS-PLOTS
# =====================================================
def create_thesis_plots(s_id, file_paths):
    if s_id not in SERIES_CONFIG:
        print(f"-> Überspringe Serie {s_id} (nicht in SERIES_CONFIG).")
        return

    print(f"\n>>> Erstelle kombinierten Plot für Serie {s_id}...")
    
    config = SERIES_CONFIG[s_id]
    original_slice_idx = config["slice_idx"]
    vis_p_low, vis_p_high = config["vis_p"]
    
    mus, sigmas = [], []
    raw_input, ground_truth = None, None
    
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        mus.append(data['pred'])
        sigmas.append(data['sigma'])
        
        if idx == 0:
            raw_input = data['lc']  
            ground_truth = data['gt'] 

    # Slicing: Entferne Ränder (0,1 und 39,40)
    mus = np.stack(mus)[:, 2:-2, :, :]
    sigmas = np.stack(sigmas)[:, 2:-2, :, :]
    raw_input = raw_input[2:-2, :, :]
    ground_truth = ground_truth[2:-2, :, :]
    
    # Berechne den neuen Z-Index aufgrund des Beschnitts (2:-2)
    z = original_slice_idx - 2 

    if z < 0 or z >= mus.shape[1]:
        print(f"Warnung: Der berechnete Slice {z} liegt außerhalb des Arrays. Überspringe.")
        return

    # Metriken berechnen
    mu_ens = np.mean(mus, axis=0)
    sigma_aleatoric = np.sqrt(np.mean(sigmas**2, axis=0))
    sigma_epistemic = np.std(mus, axis=0)
    sigma_relative = sigma_epistemic / (mu_ens + 1e-6)

    # ---------------------------------------------------------
    # DER 2x3 KOMBINIERTE PLOT
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=300)
    fig.suptitle(f"Uncertainty Decomposition - Series {s_id} | Slice {original_slice_idx}", fontsize=22, y=1.02)
    
    # Definition der Inhalte für die OBERE REIHE (Keine Colorbars, invertiertes Grau)
    images_top = [raw_input[z], ground_truth[z], mu_ens[z]]
    titles_top = ["Low Count (Input)", "Ground Truth", "Ensemble $\mu_{ens}$"]
    
    for i in range(3):
        ax = axes[0, i]
        # Bild normieren mit deinen perfekten vis_p Werten
        img_norm = vis_norm(images_top[i], vis_p_low, vis_p_high)
        ax.imshow(img_norm, cmap='gray_r')
        ax.set_title(titles_top[i], fontsize=18)
        # Nur ganz links die Y-Achse beschriften
        if i == 0: 
            ax.set_ylabel("Detector Y", fontsize=14)

# Definition der Inhalte für die UNTERE REIHE (Mit Colorbars, Inferno Colormap)
    images_bottom = [sigma_aleatoric[z], sigma_epistemic[z], sigma_relative[z]]
    titles_bottom = ["Aleatoric Uncertainty", "Epistemic Uncertainty (Disagreement)", "Relative Epistemic Uncertainty"]
    
    for i in range(3):
        ax = axes[1, i]
        
        # Anti-Überbelichtungs-Fix für die Aleatoric Uncertainty (Index 0)
        if i == 0:
            # 99.9% nimmt fast das absolute Maximum, das dunkelt das Bild schön ab
            vmax_val = np.percentile(images_bottom[i], 99.0) 
        else:
            # Für die anderen beiden nutzen wir weiterhin deine perfekte Config
            vmax_val = np.percentile(images_bottom[i], vis_p_high)
            
        im = ax.imshow(images_bottom[i], cmap='inferno', vmin=0, vmax=vmax_val)
        ax.set_title(titles_bottom[i], fontsize=18)
        ax.set_xlabel("Detector X", fontsize=14)
        if i == 0: 
            ax.set_ylabel("Detector Y", fontsize=14)
        
        # Colorbar anfügen
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"Plot_Combined_S{s_id}.png", bbox_inches='tight')
    plt.close(fig)

    print(f"   -> 2x3 Plot erfolgreich gespeichert!")

# =====================================================
# 4. RUNNER LOGIK
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR.glob("*.npz")))
    ensembles = defaultdict(list)

    target_series = [s for s in [5, 12, 13] if s in SERIES_CONFIG]

    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            if s_id in target_series:
                ensembles[s_id].append(f)

    for s_id, file_paths in ensembles.items():
        if len(file_paths) == 10:
            create_thesis_plots(s_id, file_paths)
        else:
            print(f"Warnung: Serie {s_id} hat {len(file_paths)} Seeds statt 10. Überspringe...")

    print("\n>>> Alle Master-Thesis Plots wurden erfolgreich erstellt!")