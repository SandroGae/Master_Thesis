#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
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

OUT_DIR_BASE = BASE_DIR / "Masked_Videos"
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

USE_MEDIAN = True
MASK_PARAMS_MU = {"GT_MASK_THRESHOLD": 0.26, "MIN_VALUE": 0.15}

SERIES_CONFIG = {
    # Wir behalten die Config, filtern aber in der Runner-Logik
    5:  {"slice_idx": 14, "vis_p": (0.5, 97.5)}, 
    11: {"slice_idx": 19, "vis_p": (0.5, 97.5)},
    12: {"slice_idx": 17, "vis_p": (0.5, 98.0)}, 
    13: {"slice_idx": 7,  "vis_p": (0.5, 88.0)},
    15: {"slice_idx": 18, "vis_p": (0.5, 98.5)}, 
    16: {"slice_idx": 16, "vis_p": (0.5, 97.5)},
    17: {"slice_idx": 17, "vis_p": (0.5, 95.0)}, 
    22: {"slice_idx": 16, "vis_p": (0.5, 98.0)},
    29: {"slice_idx": 24, "vis_p": (0.5, 95.0)}, 
    30: {"slice_idx": 14, "vis_p": (0.5, 97.0)},
    32: {"slice_idx": 31, "vis_p": (0.5, 92.0)}, 
    35: {"slice_idx": 23, "vis_p": (0.5, 90.0)},
    36: {"slice_idx": 15, "vis_p": (0.5, 95.0)}, 
    38: {"slice_idx": 35, "vis_p": (0.5, 90.0)},
    41: {"slice_idx": 18, "vis_p": (0.5, 94.5)}, 
    42: {"slice_idx": 35, "vis_p": (0.5, 90.0)},
    45: {"slice_idx": 20, "vis_p": (0.5, 97.0)}, 
    46: {"slice_idx": 37, "vis_p": (0.5, 90.0)},
    50: {"slice_idx": 12, "vis_p": (0.5, 96.0)}, 
    51: {"slice_idx": 22, "vis_p": (0.5, 94.0)},
    55: {"slice_idx": 21, "vis_p": (0.5, 95.0)}, 
    56: {"slice_idx": 9,  "vis_p": (0.5, 95.0)},
    57: {"slice_idx": 22, "vis_p": (0.5, 98.0)}, 
    59: {"slice_idx": 15, "vis_p": (0.5, 96.0)},
    64: {"slice_idx": 5,  "vis_p": (0.5, 95.0)}, 
    67: {"slice_idx": 10, "vis_p": (0.5, 95.0)},
    68: {"slice_idx": 20, "vis_p": (0.5, 95.0)}, 
    72: {"slice_idx": 15, "vis_p": (0.5, 94.0)},
    73: {"slice_idx": 25, "vis_p": (0.5, 95.0)}, 
    74: {"slice_idx": 23, "vis_p": (0.5, 96.0)},
}

# =====================================================
# 2. CORE FUNCTIONS
# =====================================================
def create_thesis_video(s_id, file_paths, p_id):
    if s_id not in SERIES_CONFIG: return
    config = SERIES_CONFIG[s_id]
    vis_p_low, vis_p_high = config["vis_p"]
    
    model_out_dir = OUT_DIR_BASE / p_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    mus, ground_truth_vol = [], None
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        mus.append(data['pred'])
        if idx == 0: ground_truth_vol = data['gt'] 

    mus = np.stack(mus)[:, 2:-2, :, :]
    ground_truth_vol = ground_truth_vol[2:-2, :, :] 
    mu_vol = np.mean(mus, axis=0)
    Z_slices, H_orig, W_orig = mu_vol.shape

    # GLOBALE Helligkeits-Limits
    vmin_global, vmax_global = np.percentile(mu_vol, [vis_p_low, vis_p_high])
    def norm_global(img): 
        return np.clip((img - vmin_global) / (vmax_global - vmin_global + 1e-8), 0, 1)

    # =========================================================
    # NEU: STATISCHE MASKE VOR DEM VIDEO BERECHNEN (Performance!)
    # =========================================================
    gt_ref = np.median(ground_truth_vol, axis=0) if USE_MEDIAN else np.mean(ground_truth_vol, axis=0)
    gt_min = np.min(ground_truth_vol, axis=0)
    
    base_mask = (gt_ref > MASK_PARAMS_MU["GT_MASK_THRESHOLD"]) & (gt_min >= MASK_PARAMS_MU["MIN_VALUE"])
    dilated_mask = ndi.binary_dilation(base_mask, structure=np.ones((3, 3)))
    closed_mask = ndi.binary_closing(dilated_mask, structure=np.ones((7, 7)))
    
    background = ~closed_mask
    labeled_bg, num_features = ndi.label(background)
    
    border_labels = set(np.unique(labeled_bg[0, :])) | set(np.unique(labeled_bg[-1, :])) | \
                    set(np.unique(labeled_bg[:, 0])) | set(np.unique(labeled_bg[:, -1]))
    
    final_mask = np.copy(closed_mask)
    for label_idx in range(1, num_features + 1):
        if label_idx not in border_labels:
            hole_mask = (labeled_bg == label_idx)
            if np.sum(hole_mask) <= 10:
                final_mask[hole_mask] = True

    exclusion_zone = ndi.binary_dilation(final_mask, structure=np.ones((10, 10)))
    
    border_mask = np.zeros((H_orig, W_orig), dtype=bool)
    border_mask[:3, :] = True; border_mask[-3:, :] = True
    border_mask[:, :3] = True; border_mask[:, -3:] = True
    # =========================================================

    # --- PLOTTING SETUP ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    
    im0 = axes[0].imshow(np.zeros((H_orig-8, W_orig-8)), cmap='gray_r', vmin=0, vmax=1)
    im1 = axes[1].imshow(np.zeros((H_orig-8, W_orig-8)), cmap='gray_r', vmin=0, vmax=1)
    im2 = axes[2].imshow(np.zeros((H_orig-8, W_orig-8)), cmap='gray_r', vmin=0, vmax=1)
    
    titles = [r"Original Prediction ($\mu_{ens}$)", "Masked (Background Filled)", "Masked (Pure White)"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=18)
        ax.axis('off')

    plt.tight_layout()

    # --- UPDATE FUNKTION FÜR DAS VIDEO ---
    def update(z):
        mu_slice = mu_vol[z]
        
        img_orig = mu_slice
        img_inpainted = np.copy(mu_slice)
        img_white = np.copy(mu_slice)
        
        H, W = mu_slice.shape
        y, x = np.indices((H, W))
        
        MIN_VAL = 0.1
        MAX_VAL = 0.25
        # Intensity Mask wird Frame für Frame berechnet
        intensity_mask = (mu_slice >= MIN_VAL) & (mu_slice <= MAX_VAL)
        
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

        # --- NORMALISIERUNG ---
        n_orig = norm_global(img_orig)
        n_inpainted = norm_global(img_inpainted)
        n_white = norm_global(img_white)
        n_white[final_mask] = 0.0 

        # --- CROP FÜR DAS PLOTTING ---
        c_orig = n_orig[4:-4, 4:-4]
        c_inpainted = n_inpainted[4:-4, 4:-4]
        c_white = n_white[4:-4, 4:-4]

        im0.set_data(c_orig)
        im1.set_data(c_inpainted)
        im2.set_data(c_white)
        
        fig.suptitle(f"Mu Ensemble Masking Analysis - {p_id} S{s_id:02d} | Z-Slice: {z:02d}", fontsize=22, y=1.02)
        print(f"Bearbeite {p_id} S{s_id:02d}: Frame {z+1}/{Z_slices}", end='\r')
        
        return [im0, im1, im2]

    print(f"\nStarte Video-Generierung für {p_id} S{s_id:02d} ({Z_slices} Slices)...")
    ani = FuncAnimation(fig, update, frames=Z_slices, blit=True)
    
    save_path = model_out_dir / f"Video_{p_id}_Mu_Tripartite_S{s_id:02d}.mp4"
    writer = FFMpegWriter(fps=8, metadata=dict(artist='Thesis'), bitrate=3000)
    
    try:
        ani.save(save_path, writer=writer)
        print(f"\n✅ Video erfolgreich gespeichert: {save_path.name}")
    except Exception as e:
        print(f"\n❌ Fehler beim Speichern des Videos.\nFehlermeldung: {e}")
    finally:
        plt.close(fig)

# =====================================================
# 3. RUNNER
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR_BEST.glob("*.npz")) + list(NPZ_DIR_CARE.glob("*.npz")))
    ensembles = defaultdict(list)
    
    target_series = [5, 12, 32]
    
    for f in all_npzs:
        m = re.search(r"(P\d+).*_S(\d+)\.npz", f.name) or re.search(r"(CARE_MSE).*_S(\d+)\.npz", f.name.replace("Confidence_MSE", "CARE_MSE"))
        if m: 
            ensembles[(m.group(1), int(m.group(2)))].append(f)

    print(f"Starte tripartite Video Analyse NUR FÜR P14 (Serien: {target_series})...")
    
    for (m_id, s_id), paths in sorted(ensembles.items()):
        if m_id == "P14" and s_id in target_series: 
            if len(paths) == 10: 
                create_thesis_video(s_id, paths, m_id)
            else:
                print(f"Hinweis: {m_id} S{s_id} hat nur {len(paths)}/10 Seeds. Überspringe.")

    print(f"\n✅ Alle Video-Jobs wurden abgeschlossen.")