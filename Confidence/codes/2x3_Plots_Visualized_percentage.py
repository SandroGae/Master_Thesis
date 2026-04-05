#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import warnings
from tqdm import tqdm

# NEU: Import für die Prozent-Colorbar
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & CONFIGURATION
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR_BEST = BASE_DIR / "npz_files" / "Best_3_Points"
NPZ_DIR_CARE = BASE_DIR / "npz_files" / "CARE_10_SEEDS"
OUT_DIR_BASE = BASE_DIR / "Thesis_Plots_Combined_Final"

OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# --- INDIVIDUELLE CUTOFFS FÜR DIE ANALYSE-REIHEN (Perzentile) ---
# 0: Aleatoric, 1: Epistemic, 2: Confluence
# 3: Weighted Risk (epi), 4: Weighted Aleatoric (NEU), 5: Total Weighted (NEU)
ANALYSIS_CUTS = {
    0: (1.0, 98.0),   # Aleatoric
    1: (0.01, 99.7),  # Epistemic
    2: (0.01, 99.5),   # Uncertainty Confluence
    3: (0.01, 99.7),  # Signal-Weighted Risk (mu * epi)
    4: (1.0, 98.0),   # Weighted Aleatoric (mu * ale) - Orientiert an Aleatoric
    5: (0.01, 99.5),   # Total Weighted (mu * ale * epi) - Orientiert an Confluence
}

SERIES_CONFIG = {
    # Block 1
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
    # Block 2
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
    # Block 3
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
# 2. HILFSFUNKTIONEN
# =====================================================
def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

# =====================================================
# 3. GENERIERUNG DES THESIS-PLOTS
# =====================================================
def create_thesis_plots(s_id, file_paths, p_id):
    if s_id not in SERIES_CONFIG:
        return

    config = SERIES_CONFIG[s_id]
    original_slice_idx = config["slice_idx"]
    vis_p_low, vis_p_high = config["vis_p"]
    
    model_out_dir = OUT_DIR_BASE / p_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    mus, sigmas = [], []
    raw_input, ground_truth = None, None
    
    for idx, path in enumerate(file_paths):
        data = np.load(path)
        mus.append(data['pred'])
        sigmas.append(data['sigma'])
        if idx == 0:
            raw_input = data['lc']  
            ground_truth = data['gt'] 

    # Slicing & Ensemble Metriken
    mus = np.stack(mus)[:, 2:-2, :, :]
    sigmas = np.stack(sigmas)[:, 2:-2, :, :]
    raw_input = raw_input[2:-2, :, :]
    ground_truth = ground_truth[2:-2, :, :]
    z = original_slice_idx - 2 

    mu_ens = np.mean(mus, axis=0)
    sigma_aleatoric = np.sqrt(np.mean(sigmas**2, axis=0))
    sigma_epistemic = np.std(mus, axis=0)
    
    # Verrechnungen
    uncertainty_confluence = sigma_aleatoric * sigma_epistemic
    weighted_risk = mu_ens * sigma_epistemic
    
    # NEUE VERRECHNUNGEN FÜR REIHE 3
    weighted_aleatoric = mu_ens * sigma_aleatoric
    total_weighted_risk = mu_ens * sigma_aleatoric * sigma_epistemic

    # ---------------------------------------------------------
    # DER 3x3 KOMBINIERTE PLOT
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(24, 18), dpi=300)
    fig.suptitle(f"Ensemble Uncertainty Decomposition: {p_id} | Series {s_id:02d} | Slice {original_slice_idx}", 
                 fontsize=24, fontweight='bold', y=1.02)
    
    # REIHE 1: REKONSTRUKTION
    images_r1 = [raw_input[z], mu_ens[z], ground_truth[z]]
    titles_r1 = ["Input (Low Count)", r"Prediction ($\mu_{ensemble}$)", "Reference (Ground Truth)"]
    
    for i in range(3):
        ax = axes[0, i]
        img_norm = vis_norm(images_r1[i], vis_p_low, vis_p_high)
        ax.imshow(img_norm, cmap='gray_r')
        ax.set_title(titles_r1[i], fontsize=20, pad=10)
        ax.axis('off')

    # REIHE 2: UNSICHERHEITS-KOMPONENTEN
    images_r2 = [sigma_aleatoric[z], sigma_epistemic[z], uncertainty_confluence[z]]
    titles_r2 = [r"Data Noise ($\sigma_{aleatoric}$)", 
                 r"Model Variation ($\sigma_{epistemic}$)", 
                 r"Uncertainty Confluence ($\sigma_{ale} \cdot \sigma_{epi}$)"]
    
    for i in range(3):
        ax = axes[1, i]
        p_low, p_high = ANALYSIS_CUTS[i]
        vmin, vmax = np.percentile(images_r2[i], [p_low, p_high])
        im = ax.imshow(images_r2[i], cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(titles_r2[i], fontsize=20, pad=10)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if i in [0, 1]:
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        else:
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
        
        ax.axis('off')

    # REIHE 3: GEWICHTETES RISIKO (Komplett gefüllt)
    images_r3 = [weighted_aleatoric[z], weighted_risk[z], total_weighted_risk[z]]
    titles_r3 = [r"Weighted Aleatoric Risk ($\mu \cdot \sigma_{ale}$)", 
                 r"Signal-Weighted Risk ($\mu \cdot \sigma_{epi}$)", 
                 r"Total Weighted Risk ($\mu \cdot \sigma_{ale} \cdot \sigma_{epi}$)"]
    cuts_r3 = [4, 3, 5]  # Index in ANALYSIS_CUTS
    
    for i in range(3):
        ax = axes[2, i]
        p_low, p_high = ANALYSIS_CUTS[cuts_r3[i]]
        vmin, vmax = np.percentile(images_r3[i], [p_low, p_high])
        im = ax.imshow(images_r3[i], cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_title(titles_r3[i], fontsize=20, pad=10)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Keine Prozente für Risiko-Plots, da Multiplikation mit mu
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_offset_position('left')
        
        ax.axis('off')

    plt.tight_layout()
    save_path = model_out_dir / f"Plot_{p_id}_Analysis_3x3_S{s_id:02d}.png"
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. RUNNER LOGIK
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR_BEST.glob("*.npz")) + list(NPZ_DIR_CARE.glob("*.npz")))
    ensembles = defaultdict(list)
    target_series = list(SERIES_CONFIG.keys()) 

    for f in all_npzs:
        s_id, m_id = None, None
        match_p = re.search(r"(P\d+).*_S(\d+)\.npz", f.name)
        if match_p:
            m_id, s_id = match_p.group(1), int(match_p.group(2))
        match_care = re.search(r"Confidence_MSE.*_S(\d+)\.npz", f.name)
        if match_care:
            m_id, s_id = "CARE_MSE", int(match_care.group(1))
            
        if m_id and s_id and s_id in target_series:
            ensembles[(m_id, s_id)].append(f)

    for (m_id, s_id), file_paths in tqdm(ensembles.items()):
        if len(file_paths) == 10:
            create_thesis_plots(s_id, file_paths, m_id)

    print(f"\n✅ Alle 3x3 Plots wurden unter {OUT_DIR_BASE} erstellt.")