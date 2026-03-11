#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
EXP_NAME = "CARE_10_SEEDS"

# Basis-Pfad exakt auf dein Windows-System angepasst
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR = BASE_DIR / "npz_files" / EXP_NAME
OUT_DIR = BASE_DIR / "Plots_Confidence_Ensemble" / EXP_NAME

# Erstellt den Ordner automatisch, falls er noch nicht da ist
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    return np.clip((image - vmin) / (max(1e-9, vmax - vmin)), 0, 1)

# =====================================================
# 2. ENSEMBLE PLOTTING FUNKTION
# =====================================================
def plot_ensemble_comparison(ensemble_key, s_id, file_paths):
    try:
        if s_id not in SERIES_CONFIG:
            return False

        cfg = SERIES_CONFIG[s_id]
        slice_z = cfg["slice_idx"]
        p_low, p_high = cfg.get("vis_p", (0.5, 99.5))

        mus = []
        sigmas = []

        # Alle 10 Seeds laden und extrahieren
        for path in file_paths:
            data = np.load(path)
            mus.append(data['pred'][slice_z])
            sigmas.append(data['sigma'][slice_z])

        # Listen in Arrays umwandeln (Shape: 10, 192, 240)
        mus = np.stack(mus)
        sigmas = np.stack(sigmas)

        # --- DIE 3 KANÄLE BERECHNEN ---
        # 1. Ensemble Mean (Die beste Rekonstruktion)
        mu_ens = np.mean(mus, axis=0)

        # 2. Aleatorische Unsicherheit (Grundrauschen der Daten)
        # Mathematisch korrekt: Varianzen mitteln, dann Wurzel ziehen
        sigma_aleatoric = np.sqrt(np.mean(sigmas**2, axis=0))

        # 3. Epistemische Unsicherheit / Disagreement (Modell-Uneinigkeit)
        # Die Standardabweichung zwischen den 10 verschiedenen Mu-Vorhersagen
        sigma_epistemic = np.std(mus, axis=0)

        # --- VISUALISIERUNG ---
        # Breiteres Layout für 3 Plots nebeneinander
        fig, axes = plt.subplots(1, 3, figsize=(28, 9), dpi=200)

        # Plot 1: Ensemble Mu
        im0 = axes[0].imshow(vis_norm(mu_ens, p_low, p_high), cmap='gray_r', aspect='equal')
        axes[0].set_title(f"Ensemble Reconstruction ($\\mu_{{ens}}$) - Serie {s_id}\n{len(file_paths)} Seeds | Slice: {slice_z} | Helligkeit: {p_low}-{p_high}%", fontsize=14)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot 2: Aleatorische Unsicherheit (Analog zu deinem bisherigen mittleren Plot)
        al_vmax = max(0.01, np.percentile(sigma_aleatoric, 99.0))
        im1 = axes[1].imshow(sigma_aleatoric, cmap='inferno', aspect='equal', vmin=0, vmax=al_vmax)
        axes[1].set_title(f"Aleatoric Uncertainty ($\\sigma_{{data}}$) - Serie {s_id}\nAverage Data Noise | Max: {al_vmax:.3f}", fontsize=14)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot 3: Epistemische Unsicherheit (Disagreement)
        ep_vmax = max(0.001, np.percentile(sigma_epistemic, 99.0))
        im2 = axes[2].imshow(sigma_epistemic, cmap='inferno', aspect='equal', vmin=0, vmax=ep_vmax)
        axes[2].set_title(f"Epistemic Uncertainty (Disagreement) - Serie {s_id}\nModel Variance | Max: {ep_vmax:.4f}", fontsize=14)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel("Detector X")
            ax.set_ylabel("Detector Y")

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Ensemble_{ensemble_key}.png", bbox_inches='tight')
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Fehler bei Ensemble {ensemble_key}: {e}")
        return False

# =====================================================
# 3. RUNNER & GRUPPIERUNG
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR.glob("*.npz")))
    print(f"Gefunden: {len(all_npzs)} NPZ-Dateien.")

    # Dictionary zum Gruppieren der 10 Seeds pro Punkt/Serie
    ensembles = defaultdict(list)

    for f in all_npzs:
        # Extrahiert die Seriennummer am Ende (z.B. "_S05.npz")
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            
            # Wir ersetzen das spezifische "seedXX" durch den Platzhalter "ENSEMBLE"
            # Beispiel: "Eval_Confidence_MSE_seed42_S05" wird zu "Eval_Confidence_MSE_ENSEMBLE_S05"
            # Dadurch landen alle 10 Seeds dieses Punktes im selben Key.
            ensemble_key = re.sub(r'seed\d+', 'ENSEMBLE', f.stem)
            
            ensembles[(ensemble_key, s_id)].append(f)

    print(f"Erstelle {len(ensembles)} Ensemble-Plots (jeweils aggregiert aus mehreren Seeds)...")
    
    for (ensemble_key, s_id), file_paths in tqdm(ensembles.items()):
        if len(file_paths) != 10:
            print(f"-> Warnung: {ensemble_key} hat {len(file_paths)} Seeds (erwartet: 10). Wird trotzdem geplottet.")
            
        plot_ensemble_comparison(ensemble_key, s_id, file_paths)

    print(f"\n>>> Fertig! Die Ensemble-Bilder liegen in: {OUT_DIR}")