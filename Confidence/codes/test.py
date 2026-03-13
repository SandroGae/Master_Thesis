import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from tqdm import tqdm

# =====================================================
# 1. SETUP & PFADE
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR_BEST = BASE_DIR / "npz_files" / "Best_3_Points"
NPZ_DIR_CARE = BASE_DIR / "npz_files" / "CARE_10_SEEDS"
OUT_DIR = BASE_DIR / "Thesis_Plots_Ensemble_Averaged"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- NEUE PARAMETER FÜR DEN TEMPORAL FILTER ---
# Ab wie viel Intensität über dem Median werten wir ein Pixel als "aktiv/leuchtend"?
TRANSIENT_ACT_THRESH = 0.05 
# In wie vielen Slices darf das Pixel MAXIMAL leuchten, um noch als CDW zu gelten?
MAX_ACTIVE_FRAMES = 8 

# Deine exakte Konfiguration!
SERIES_CONFIG = {
    # Block 1
    5:  {"slice_idx": 14, "vis_p": (0.5, 97.5)},
    11: {"slice_idx": 19, "vis_p": (0.5, 97.5)},
    12: {"slice_idx": 17, "vis_p": (0.5, 98.0)},
    13: {"slice_idx": 7,  "vis_p": (0.5, 88.0)},
}
"""
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
"""

def vis_norm(image, p_low, p_high):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

# =====================================================
# 2. DATEIEN SUCHEN & GRUPPIEREN
# =====================================================
print("Suche nach NPZ-Dateien...")
all_files = list(NPZ_DIR_BEST.glob("*.npz")) + list(NPZ_DIR_CARE.glob("*.npz"))

grouped_files = defaultdict(lambda: defaultdict(list))

for f in all_files:
    match = re.search(r"Eval_(.+)_seed(\d+)_S(\d+)\.npz", f.name)
    if match:
        model_name = match.group(1)
        series_id = int(match.group(3))
        if series_id in SERIES_CONFIG:
            grouped_files[model_name][series_id].append(f)

# =====================================================
# 3. VERARBEITUNG & PLOTTING
# =====================================================
for model_name, series_dict in grouped_files.items():
    print(f"\n>>> Erstelle Plots für Modell: {model_name}")
    
    model_out_dir = OUT_DIR / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    
    for s_id, file_paths in tqdm(series_dict.items(), desc="Serien verarbeiten"):
        if len(file_paths) != 10:
            pass 
            
        config = SERIES_CONFIG[s_id]
        z_idx = config["slice_idx"]
        p_low, p_high = config["vis_p"]
        
        # 1. Ensemble Averaging über das ganze Volumen
        preds, sigmas = [], []
        for path in file_paths:
            data = np.load(path)
            preds.append(data['pred'])
            sigmas.append(data['sigma'])
            
        mu_ens_vol = np.mean(preds, axis=0)
        sigma_ens_vol = np.mean(sigmas, axis=0)
        if mu_ens_vol.ndim == 4: mu_ens_vol = mu_ens_vol[..., 0]
        if sigma_ens_vol.ndim == 4: sigma_ens_vol = sigma_ens_vol[..., 0]
        
        # 2. Slices extrahieren
        mu = mu_ens_vol[z_idx]
        sigma = sigma_ens_vol[z_idx]
        
        # 3. NEU: Plot 3 (Uncertainty-Weighted Signal)
        weighted_signal = mu * sigma
        
        # 4. NEU: Plot 4 (Spatio-Temporal Masking)
        # a) Finde den statischen Hintergrund (Median über alle Bilder)
        temporal_median = np.median(mu_ens_vol, axis=0)
        
        # b) In wie vielen Frames leuchtet der Pixel signifikant heller als sein eigener Median?
        is_active = mu_ens_vol > (temporal_median + TRANSIENT_ACT_THRESH)
        active_frames_count = np.sum(is_active, axis=0)
        
        # c) Erstelle die Transient-Maske (Muss in mind. 1 Frame leuchten, darf aber nicht Dauer-Flackern)
        transient_mask = (active_frames_count > 0) & (active_frames_count <= MAX_ACTIVE_FRAMES)
        
        # d) Signal bereinigen: Alles was statisch/flackernd ist, wird auf 0 gesetzt!
        temporal_filtered_signal = np.copy(weighted_signal)
        temporal_filtered_signal[~transient_mask] = 0.0
        
        # 5. Plotten
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
        fig.patch.set_facecolor('white')
        fig.suptitle(f"Model: {model_name} | Series {s_id} | Slice {z_idx}", fontsize=18)
        
        # Plot 1: Ensemble Durchschnitt
        im0 = axes[0].imshow(vis_norm(mu, p_low, p_high), cmap='magma')
        axes[0].set_title('1. Ensemble Rekonstruktion ($\mu_{ens}$)', fontsize=14)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2: Ensemble Unsicherheit
        vmax_sigma = np.percentile(sigma, 99.5)
        im1 = axes[1].imshow(sigma, cmap='inferno', vmin=0, vmax=vmax_sigma)
        axes[1].set_title(f'2. Ensemble Uncertainty ($\sigma_{{ens}}$)', fontsize=14)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot 3: Das gewichtete Signal (mu * sigma)
        # Wir normieren von 0 an, damit der Hintergrund schön schwarz bleibt
        im2 = axes[2].imshow(vis_norm(weighted_signal, 0.0, 99.5), cmap='magma')
        axes[2].set_title('3. Uncertainty-Weighted Signal ($\mu \cdot \sigma$)\n(Unterdrückt sicheres Rauschen)', fontsize=14)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Plot 4: Temporal Spatio-Filtering
        im3 = axes[3].imshow(vis_norm(temporal_filtered_signal, 0.0, 99.5), cmap='magma')
        axes[3].set_title(f'4. Spatio-Temporal Filter\n(Löscht statisches/flackerndes Signal > {MAX_ACTIVE_FRAMES} Slices)', fontsize=14)
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        
        # Plot speichern
        save_name = f"{model_name}_Series_{s_id:02d}_Slice_{z_idx}.png"
        fig.savefig(model_out_dir / save_name, bbox_inches='tight')
        plt.close(fig)

print("\n>>> Alle Plots wurden erfolgreich generiert und gespeichert!")