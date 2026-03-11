#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
from collections import defaultdict
import io
import imageio.v3 as iio
import warnings
import scipy.ndimage as ndi

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
EXP_NAME = "CARE_10_SEEDS"

# Basis-Pfad exakt auf dein Windows-System angepasst
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")
NPZ_DIR = BASE_DIR / "npz_files" / EXP_NAME
OUT_DIR = BASE_DIR / "Plots_Confidence_Ensemble_Movies" / EXP_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wir wollen nur bestimmte Serien
TARGET_SERIES = [5, 11, 12, 13, 15, 16, 17, 22, 29, 30, 32, 35, 36, 38, 41, 42, 45, 46, 50, 51, 55, 56, 57, 59, 64, 67,68, 72, 73, 74]
FPS = 2

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    return np.clip((image - vmin) / (max(1e-9, vmax - vmin)), 0, 1)


def isolate_cdw_kinematics(sigma_relative, mu_ens):
    """
    Ultimativer Hybrid-Filter: Y-Band Scout + Strenge Morphologie + Hartes Mu-Veto.
    """
    print("-> Starte Scout-Band + Strenge Morphologie + Mu-Veto...")

    isolated_cdw_signal = np.zeros_like(sigma_relative)
    num_slices = sigma_relative.shape[0]

    # =========================================================
    # PHASE 1: SCOUT (Finde die exakte Y-Höhe der CDW)
    # =========================================================
    strict_y_centers = []

    for z in range(num_slices):
        sig_frame = sigma_relative[z]
        smoothed = ndi.gaussian_filter(sig_frame, sigma=(0.5, 3.0))
        tophat = ndi.white_tophat(smoothed, footprint=np.ones((14, 1)))

        thresh = np.percentile(tophat, 98.5)
        binary_mask = tophat > thresh
        labeled, num_features = ndi.label(binary_mask)

        for i in range(1, num_features + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
            x_min, x_max = coords[:, 1].min(), coords[:, 1].max()

            h = (y_max - y_min) + 1
            w = (x_max - x_min) + 1

            # EXTREM STRENG: Nur perfekte, lange CDWs dürfen die Höhe vorgeben
            if w >= 25 and h <= 12 and (w / h) >= 2.5:
                y_center = (y_min + y_max) / 2.0
                strict_y_centers.append(y_center)

    if len(strict_y_centers) == 0:
        print("   WARNUNG: Keine klare CDW-Spur gefunden. Breche ab für diese Serie.")
        return isolated_cdw_signal

    best_y = int(np.median(strict_y_centers))
    y_start = max(0, best_y - 8)
    y_end = min(sigma_relative.shape[1], best_y + 8)
    print(f"   CDW-Spur fixiert auf Y={best_y}. Wende strenge Filter in Zone [{y_start}, {y_end}] an.")

    # =========================================================
    # PHASE 2: STRICT FILTERING (Nur in der CDW-Zone!)
    # =========================================================
    valid_blobs = 0

    for z in range(num_slices):
        sig_frame = sigma_relative[z]
        mu_frame = mu_ens[z]

        band_sig = sig_frame[y_start:y_end, :]
        band_mu = mu_frame[y_start:y_end, :]

        # Glätten und Top-Hat (Nur im Schlauch, schneidet dicke Artefakte auf der Spur durch)
        smoothed = ndi.gaussian_filter(band_sig, sigma=(0.5, 4.0))
        tophat = ndi.white_tophat(smoothed, footprint=np.ones((14, 1)))

        # Binarisierung, um die Blobs zu finden
        thresh = np.percentile(tophat, 97.0)
        binary_mask = tophat > thresh
        labeled, num_features = ndi.label(binary_mask)

        for i in range(1, num_features + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)

            y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
            x_min, x_max = coords[:, 1].min(), coords[:, 1].max()

            h = (y_max - y_min) + 1
            w = (x_max - x_min) + 1

            # 1. STRENGE GEOMETRIE: Es MUSS ein klarer horizontaler Strich sein!
            # Das blockt die kleinen punktförmigen Artefakte auf dieser Y-Höhe ab.
            if w >= 18 and h <= 12 and (w / h) >= 2.0:

                # 2. STRENGES MU-VETO: Muss im physikalischen mu_ens Kanal deutlich sichtbar sein!
                # Wir verlangen, dass der hellste Pixel dieses Blob-Kandidaten zu den 
                # absolut hellsten 15 % des GESAMTEN Bildes gehört.
                mu_bg_level = np.percentile(mu_frame, 85.0)

                if np.max(band_mu[mask]) > mu_bg_level:
                    # TREFFER! Wir schreiben die echten Unsicherheitswerte zurück ins Bild.
                    isolated_cdw_signal[z, y_start:y_end, :][mask] = band_sig[mask]
                    valid_blobs += 1

    print(f"   {valid_blobs} echte CDW-Blobs nach Morphologie und Mu-Veto gerettet.")
    return isolated_cdw_signal



def create_ensemble_movie(s_id, file_paths):
    print(f"\n>>> Lade {len(file_paths)} Seeds für Serie {s_id}...")
    
    mus, sigmas = [], []
    for path in file_paths:
        data = np.load(path)
        mus.append(data['pred'])
        sigmas.append(data['sigma'])

    mus = np.stack(mus)
    sigmas = np.stack(sigmas)
    num_slices = mus.shape[1]
    
    print(f"-> Berechne Ensemble-Metriken über {num_slices} Slices...")
    mu_ens = np.mean(mus, axis=0)
    sigma_aleatoric = np.sqrt(np.mean(sigmas**2, axis=0))
    sigma_epistemic = np.std(mus, axis=0)
    
    # Video 4 Logik bleibt exakt so
    sigma_relative = sigma_epistemic / (mu_ens + 1e-6)
    
    # Video 5 Logik anwenden
    sigma_cdw_isolated = isolate_cdw_kinematics(sigma_relative, mu_ens)

    print("-> Ermittle globale Skalierungswerte...")
    al_vmax = max(0.01, np.percentile(sigma_aleatoric, 99.5))
    ep_vmax = max(0.001, np.percentile(sigma_epistemic, 99.5))
    rel_vmax = max(0.01, np.percentile(sigma_relative, 99.5))
    iso_vmax = max(0.001, np.percentile(sigma_cdw_isolated, 99.9)) 
    
    frames = []
    print(f"-> Rendere {num_slices} Frames für Video im 2x3 Layout...")
    
    for z in tqdm(range(num_slices)):
        # 2x3 Layout definieren
        fig, axes = plt.subplots(2, 3, figsize=(24, 16), dpi=120)
        
        # --- OBERE REIHE (3 Plots) ---
        # 1. Ensemble Mu
        im0 = axes[0, 0].imshow(vis_norm(mu_ens[z], 0.5, 98.0), cmap='gray_r', aspect='equal')
        axes[0, 0].set_title(f"Ensemble Reconstruction ($\\mu_{{ens}}$) - S{s_id} | Slice: {z}", fontsize=14)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Aleatorische Unsicherheit
        im1 = axes[0, 1].imshow(sigma_aleatoric[z], cmap='inferno', aspect='equal', vmin=0, vmax=al_vmax)
        axes[0, 1].set_title(f"Aleatoric Uncertainty ($\\sigma_{{data}}$)\nAverage Data Noise", fontsize=14)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. Epistemische Unsicherheit
        im2 = axes[0, 2].imshow(sigma_epistemic[z], cmap='inferno', aspect='equal', vmin=0, vmax=ep_vmax)
        axes[0, 2].set_title(f"Epistemic Uncertainty (Disagreement)\nModel Variance", fontsize=14)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # --- UNTERE REIHE (2 Plots) ---
        # 4. Relative Epistemische Unsicherheit
        im3 = axes[1, 0].imshow(sigma_relative[z], cmap='inferno', aspect='equal', vmin=0, vmax=rel_vmax)
        axes[1, 0].set_title(f"Relative Uncertainty\nDisagreement / Signal", fontsize=14)
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 5. Isolierte CDW (Höhenfilter)
        im4 = axes[1, 1].imshow(sigma_cdw_isolated[z], cmap='inferno', aspect='equal', vmin=0, vmax=iso_vmax)
        axes[1, 1].set_title(f"Isolated CDW\nMulti-Channel Veto Filter", fontsize=14)
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 6. Das letzte Panel unsichtbar machen
        axes[1, 2].axis('off')

        # Achsen-Labels nur für die aktiven 5 Plots
        active_axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
        for ax in active_axes:
            ax.set_xlabel("Detector X")
            ax.set_ylabel("Detector Y")

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(iio.imread(buf))
        plt.close(fig)

    out_path = OUT_DIR / f"Ensemble_Movie_All_Steps_S{s_id}.mp4"
    print(f"-> Speichere Video unter {out_path}...")
    iio.imwrite(out_path, frames, fps=FPS, extension=".mp4", macro_block_size=None)

# =====================================================
# 3. RUNNER LOGIK
# =====================================================
if __name__ == "__main__":
    all_npzs = sorted(list(NPZ_DIR.glob("*.npz")))
    ensembles = defaultdict(list)

    for f in all_npzs:
        match = re.search(r"_S(\d+)\.npz", f.name)
        if match:
            s_id = int(match.group(1))
            if s_id in TARGET_SERIES:
                ensembles[s_id].append(f)

    for s_id, file_paths in ensembles.items():
        if len(file_paths) == 10:
            create_ensemble_movie(s_id, file_paths)
        else:
            print(f"Warnung: Serie {s_id} hat {len(file_paths)} Seeds statt 10. Überspringe...")

    print("\n>>> Alle Videos wurden erfolgreich erstellt!")