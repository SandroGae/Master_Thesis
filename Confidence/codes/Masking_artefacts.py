#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import warnings
from tqdm import tqdm
import scipy.ndimage as ndi
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & CONFIGURATION
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR_BEST = BASE_DIR / "npz_files" / "Best_3_Points"
NPZ_DIR_CARE = BASE_DIR / "npz_files" / "CARE_10_SEEDS"
OUT_DIR_BASE = BASE_DIR / "Masked_Plots"
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# --- DEIN TEST-BEREICH ---
USE_MEDIAN = True

# LOGIK: Pixel wird maskiert, wenn...
# 1. Sein Median über die Serie >= GT_MASK_THRESHOLD
# 2. Er in JEDEM Bild der Serie >= MIN_VALUE ist (nie darunter fällt)
GT_MASK_THRESHOLD = 0.28 
MIN_VALUE = 0.25          

BOTTOM_ROW_CUTS = {
    0: (1.0, 98.0),   # Aleatoric
    1: (0.01, 99.7),  # Epistemic
    2: (0.1, 99.7),  # Weighted Signal
}

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
    55: {"slice_idx": 21, "roi_x": (177, 198), "roi_y": (0, 192), "bg_gap": -76,  "vis_p": (0.5, 95.0)},
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


# =====================================================
# 3. GENERIERUNG DES THESIS-PLOTS
# =====================================================
def create_thesis_plots(s_id, file_paths, p_id):
    if s_id not in SERIES_CONFIG:
        return

    config = SERIES_CONFIG[s_id]
    original_slice_idx = config["slice_idx"]
    model_out_dir = OUT_DIR_BASE / p_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    mus, sigmas = [], []
    ground_truth_vol = None
    
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        mus.append(data['pred'])
        sigmas.append(data['sigma'])
        if idx == 0:
            ground_truth_vol = data['gt'] 

    # Slicing (2:-2)
    mus = np.stack(mus)[:, 2:-2, :, :]
    sigmas = np.stack(sigmas)[:, 2:-2, :, :]
    ground_truth_vol = ground_truth_vol[2:-2, :, :] 
    
    z = original_slice_idx - 2 

    # Ensemble Metriken für Slice z
    mu_slice = np.mean(mus, axis=0)[z]
    alea_slice = np.sqrt(np.mean(sigmas**2, axis=0))[z]
    epi_slice = np.std(mus, axis=0)[z]
    weight_slice = mu_slice * epi_slice

# =====================================================
    # KOMPLEXE MASKEN-LOGIK
    # =====================================================
    # 1. Zentralwert berechnen (Wahlweise Median oder Mean)
    if USE_MEDIAN:
        gt_reference = np.median(ground_truth_vol, axis=0)
        ref_label = "Median"
    else:
        gt_reference = np.mean(ground_truth_vol, axis=0)
        ref_label = "Mean"

    # 2. Minimum über die Serie (Pixel darf nie unter MIN_VALUE fallen)
    gt_min = np.min(ground_truth_vol, axis=0)
    
    # Bedingung: (Referenz > Threshold) UND (Minimum >= MIN_VALUE)
    base_mask = (gt_reference > GT_MASK_THRESHOLD) & (gt_min >= MIN_VALUE)
    
    # 3. Erweiterung auf die 8 umliegenden Pixel
    structure = np.ones((3, 3))
    final_mask = ndi.binary_dilation(base_mask, structure=structure)
    
    num_pixels_base = np.sum(base_mask)
    print(f"\n  -> {p_id} (S{s_id}): {num_pixels_base} Pixel via {ref_label} maskiert.")
    
    num_pixels_base = np.sum(base_mask)
    num_pixels_final = np.sum(final_mask)
    
    print(f"\n  -> {p_id}: {num_pixels_base} Pixel erfüllen die Kriterien.")
    print(f"     Nach 8-Nachbarn-Erweiterung werden {num_pixels_final} Pixel maskiert.")

    # --- DATEN VORBEREITEN (Korrektur für Poisson-Rauschen auf Float-Werten) ---
    row1 = [alea_slice, epi_slice, weight_slice]
    titles = ["Aleatoric Uncertainty", "Epistemic Uncertainty", r"Weighted Signal ($\mu_{ens} \cdot \sigma_{epi}$)"]

    row2 = []
    for img in row1:
        masked_img = np.copy(img)
        
        # 1. Koordinaten-Gitter erstellen
        y_coords, x_coords = np.indices(img.shape)
        
        # 2. "Spender-Pixel" und "Ziel-Pixel" definieren
        coords_unmasked = np.column_stack((y_coords[~final_mask], x_coords[~final_mask]))
        values_unmasked = img[~final_mask]
        coords_masked = np.column_stack((y_coords[final_mask], x_coords[final_mask]))
        
        # Überspringen, falls nichts maskiert wurde
        if len(coords_masked) == 0:
            row2.append(masked_img)
            continue
            
        # 3. KD-Tree für die Suche nach den K nächsten Nachbarn
        K = 30
        tree = KDTree(coords_unmasked)
        _, indices = tree.query(coords_masked, k=K)
        
        # 4. NEUE LOGIK: Zufälliges Sampling statt Averaging!
        # Für jeden maskierten Pixel würfeln wir, welchen der 30 Nachbarn wir kopieren
        if indices.ndim > 1:
            rand_neighbor = np.random.randint(0, K, size=len(coords_masked))
            # Wir extrahieren exakt diesen einen Nachbarn aus der Matrix
            sampled_values = values_unmasked[indices[np.arange(len(indices)), rand_neighbor]]
        else:
            sampled_values = values_unmasked[indices]
        
        # 5. Maskierte Pixel mit den echten Hintergrund-Werten füllen
        masked_img[final_mask] = sampled_values
        
        row2.append(masked_img)

    # --- PLOTTING (Innerhalb der create_thesis_plots Funktion) ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=300)
    fig.suptitle(f"Persistence Analysis (Recalculated Scaling) - {p_id} S{s_id}", fontsize=22, y=1.02)
    
    for col in range(3):
        cmap = 'inferno' if col < 2 else 'magma'
        p_low, p_high = BOTTOM_ROW_CUTS[col]
        
        # 1. Skalierung OBERE REIHE (Normal)
        v0_min, v0_max = np.percentile(row1[col], [p_low, p_high])
        im_top = axes[0, col].imshow(row1[col], cmap=cmap, vmin=v0_min, vmax=v0_max)
        axes[0, col].set_title(titles[col], fontsize=18)
        plt.colorbar(im_top, ax=axes[0, col], fraction=0.046, pad=0.04)

        # 2. Skalierung UNTERE REIHE (Neu berechnet!)
        # Wir berechnen das Perzentil NUR von den Pixeln, die NICHT maskiert wurden
        remaining_pixels = row2[col][~final_mask]
        
        if remaining_pixels.size > 0:
            v1_min, v1_max = np.percentile(remaining_pixels, [p_low, p_high])
        else:
            v1_min, v1_max = 0, 1 # Fallback, falls alles maskiert wurde

        im_bot = axes[1, col].imshow(row2[col], cmap=cmap, vmin=v1_min, vmax=v1_max)
        axes[1, col].set_title(titles[col] + "\n(Masked & Rescaled)", fontsize=18)
        plt.colorbar(im_bot, ax=axes[1, col], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    save_path = model_out_dir / f"Plot_{p_id}_ComplexMask_S{s_id:02d}.png"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. RUNNER LOGIK (Geändert für alle Serien in SERIES_CONFIG)
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR_BEST.glob("*.npz")) + list(NPZ_DIR_CARE.glob("*.npz")))
    
    # Wir gruppieren jetzt nach (Modell, Serie)
    ensembles = defaultdict(list)

    for f in all_npzs:
        s_id, m_id = None, None
        match_p = re.search(r"(P\d+).*_S(\d+)\.npz", f.name)
        if match_p:
            m_id, s_id = match_p.group(1), int(match_p.group(2))
        
        match_care = re.search(r"Confidence_MSE.*_S(\d+)\.npz", f.name)
        if match_care:
            m_id, s_id = "CARE_MSE", int(match_care.group(1))
            
        # Prüfen, ob die gefundene Serie in deiner SERIES_CONFIG Liste ist
        if s_id in SERIES_CONFIG:
            ensembles[(m_id, s_id)].append(f)

    # Die Schleife läuft jetzt über alle Modelle und alle 4 Serien
    print(f"Starte Prozess für {len(ensembles)} Kombinationen (Serie 5, 11, 12, 13)...")
    
    # Wir sortieren die Items, damit die Plots in einer logischen Reihenfolge erstellt werden
    for (m_id, s_id), file_paths in tqdm(sorted(ensembles.items())):
        if len(file_paths) == 10:
            # Wir übergeben jetzt die s_id aus dem Loop anstatt TARGET_S_ID
            create_thesis_plots(s_id, file_paths, m_id)
        else:
            print(f"Hinweis: {m_id} S{s_id} hat nur {len(file_paths)}/10 Seeds. Überspringe.")

    print(f"\n✅ Alle Plots für die Config wurden sauber unter {OUT_DIR_BASE} erstellt.")