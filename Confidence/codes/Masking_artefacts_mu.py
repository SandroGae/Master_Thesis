#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

USE_MEDIAN = True

# --- DEINE PARAMETER FÜR DIE MASKIERUNG ---
PRED_MASK_THRESHOLD = 0.26  # Vorheriger Wert: 0.26
PRED_MIN_VALUE = 0.15       # Vorheriger Wert: 0.15
# ------------------------------------------

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
# 2. CORE FUNCTIONS
# =====================================================
def create_thesis_plots(s_id, file_paths, p_id):
    if s_id not in SERIES_CONFIG: return
    config = SERIES_CONFIG[s_id]
    original_slice_idx, (vis_p_low, vis_p_high) = config["slice_idx"], config["vis_p"]
    
    model_out_dir = OUT_DIR_BASE / p_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    mus, ground_truth_vol = [], None
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        mus.append(data['pred'])
        if idx == 0: ground_truth_vol = data['gt'] 

    mus = np.stack(mus)[:, 2:-2, :, :]
    
    # 1. Ensemble Mean Volume berechnen (Summe der 10 Seeds geteilt durch 10)
    mu_vol = np.mean(mus, axis=0) 
    
    # Der finale 2D Slice für den aktuellen Plot
    z = original_slice_idx - 2 
    mu_slice = mu_vol[z]

    # --- MASKEN LOGIK BASIEREND AUF DER PREDICTION (Ensemble Mean) ---
    pred_ref = np.median(mu_vol, axis=0) if USE_MEDIAN else np.mean(mu_vol, axis=0)
    pred_min = np.min(mu_vol, axis=0)
    
    base_mask = (pred_ref > PRED_MASK_THRESHOLD) & (pred_min >= PRED_MIN_VALUE)
    
    # (3, 3) bedeutet: 3 Zeilen vertikal (Y), 3 Spalten horizontal (X)
    dilated_mask = ndi.binary_dilation(base_mask, structure=np.ones((3, 5)))
    closed_mask = ndi.binary_closing(dilated_mask, structure=np.ones((7, 7)))
    
    # NEUE LOGIK: Finde alle Löcher
    background = ~closed_mask
    labeled_bg, num_features = ndi.label(background)
    
    border_labels = set(np.unique(labeled_bg[0, :])) | set(np.unique(labeled_bg[-1, :])) | \
                    set(np.unique(labeled_bg[:, 0])) | set(np.unique(labeled_bg[:, -1]))
    
    final_mask = np.copy(closed_mask)
    
    for label_idx in range(1, num_features + 1):
        if label_idx not in border_labels:
            hole_mask = (labeled_bg == label_idx)
            hole_size = np.sum(hole_mask)
            if hole_size <= 10:
                final_mask[hole_mask] = True

    img_orig = mu_slice
    img_inpainted = np.copy(mu_slice)
    img_white = np.copy(mu_slice)
    
    # --- VERBESSERTES INPAINTING ---
    H, W = mu_slice.shape
    y, x = np.indices((H, W))
    
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:3, :] = True
    border_mask[-3:, :] = True
    border_mask[:, :3] = True
    border_mask[:, -3:] = True

    # =====================================================================
    # LÜCKE ZUM RAND SCHLIESSEN
    # =====================================================================
    mask_expansion = ndi.binary_dilation(final_mask, structure=np.ones((7, 7)))
    final_mask = final_mask | (border_mask & mask_expansion)
    # =====================================================================
    
    MIN_VAL = 0.1
    MAX_VAL = 0.25
    intensity_mask = (mu_slice >= MIN_VAL) & (mu_slice <= MAX_VAL)
    
    exclusion_zone = ndi.binary_dilation(final_mask, structure=np.ones((10, 10)))
    valid_source_mask = (~final_mask) & (~border_mask) & intensity_mask & (~exclusion_zone)
    
    c_valid = np.column_stack((y[valid_source_mask], x[valid_source_mask]))
    v_valid = mu_slice[valid_source_mask]
    c_masked = np.column_stack((y[final_mask], x[final_mask]))
    
    if len(c_masked) > 0 and len(c_valid) > 0:
        tree = KDTree(c_valid)
        K_fetch = min(50, len(c_valid))
        _, indices = tree.query(c_masked, k=K_fetch)
        if indices.ndim == 1:
            indices = indices[:, None]
            
        usage_counts = np.zeros(len(c_valid), dtype=int)
        fill_values = np.zeros(len(c_masked), dtype=mu_slice.dtype)
        process_order = np.random.permutation(len(c_masked))
        
        for idx in process_order:
            neighbors = indices[idx]
            top_10 = neighbors[:10]
            valid_top_10 = [n for n in top_10 if usage_counts[n] < 3]
            
            if valid_top_10:
                chosen_n = np.random.choice(valid_top_10)
            else:
                valid_fallback = [n for n in neighbors if usage_counts[n] < 3]
                if valid_fallback:
                    chosen_n = valid_fallback[0]
                else:
                    free_anywhere = np.where(usage_counts < 3)[0]
                    if len(free_anywhere) > 0:
                        chosen_n = np.random.choice(free_anywhere)
                    else:
                        chosen_n = neighbors[0]
                        
            usage_counts[chosen_n] += 1
            fill_values[idx] = v_valid[chosen_n]
            
        img_inpainted[final_mask] = fill_values

    # --- NORMALISIERUNG & FIX ---
    vmin, vmax = np.percentile(mu_slice, [vis_p_low, vis_p_high])
    def norm(img): return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)

    n_orig = norm(img_orig)
    n_inpainted = norm(img_inpainted)
    n_white = norm(img_white)
    
    n_white[final_mask] = 0.0 # (1.0 macht es "Pure White", 0.0 wäre schwarz)

    # =========================================================
    # BILDER CROPPEN (Den 3-Pixel Rand überall abschneiden)
    # =========================================================
    n_orig = n_orig[3:-3, 3:-3]
    n_inpainted = n_inpainted[3:-3, 3:-3]
    n_white = n_white[3:-3, 3:-3]

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(f"Mu Ensemble Masking Analysis - {p_id} S{s_id:02d}", fontsize=22, y=1.02)
    
    titles = [r"Original Prediction ($\mu_{ens}$)", "Masked (Background Filled)", "Masked (Pure White)"]
    imgs = [n_orig, n_inpainted, n_white]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(titles[i], fontsize=18)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(model_out_dir / f"Plot_{p_id}_Mu_Tripartite_S{s_id:02d}.png", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR_BEST.glob("*.npz")) + list(NPZ_DIR_CARE.glob("*.npz")))
    ensembles = defaultdict(list)
    
    for f in all_npzs:
        m = re.search(r"(P\d+).*_S(\d+)\.npz", f.name) or re.search(r"(CARE_MSE).*_S(\d+)\.npz", f.name.replace("Confidence_MSE", "CARE_MSE"))
        if m: 
            ensembles[(m.group(1), int(m.group(2)))].append(f)

    print(f"Starte tripartite Analyse NUR FÜR P14...")
    
    for (m_id, s_id), paths in tqdm(sorted(ensembles.items())):
        if m_id == "P14": 
            if len(paths) == 10: 
                create_thesis_plots(s_id, paths, m_id)
            else:
                print(f"Hinweis: {m_id} S{s_id} hat nur {len(paths)}/10 Seeds. Überspringe.")

    print(f"\n✅ Alle tripartite Plots für P14 wurden unter {OUT_DIR_BASE} erstellt.")