#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# Konfiguration
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "visualisation"


# Format: (Serien_Nummer, Bild_Nummer) 1 basiert
VIS_PAIRS = [
    (5, 15), (11, 20), (12, 18), (13, 1), (15, 19),
    (16, 17), (21, 19), (22, 17), (29, 25), (50, 13)
]

SERIES_LEN = 41


# Hilfsfunktionen
def normalize_image(image, scale=10000.0):
    """ Slicewise Sum-Normalization (wie im Training) """
    data = np.maximum(image, 0.0).astype(np.float32)
    s = np.sum(data) + 1e-12
    return (data / s) * scale

def vis_scaling(image):
    """ Perzentil-Scaling für optimale Sichtbarkeit """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)




def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starte Visualisierung der {len(VIS_PAIRS)} Referenzbilder...")

    with h5py.File(H5_TEST_PATH, "r") as f:
        for s_idx, img_idx in VIS_PAIRS:
            # Berechnung des globalen Index (0-basiert)
            global_idx = (s_idx - 1) * SERIES_LEN + (img_idx - 1)
            
            # Daten laden (Ground Truth)
            # H5 Shape ist meist (H, W, N) -> wir brauchen [:, :, global_idx]
            gt_raw = f["high_count/data"][:, :, global_idx]
            
            # Normalisieren & Scalen
            gt_norm = normalize_image(gt_raw)
            gt_plot = vis_scaling(gt_norm)
            
            # Plot erstellen
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
            im = ax.imshow(gt_plot, cmap="gray_r", origin="upper")
            
            # Achsen beschriften für einfaches Ablesen der Koordinaten
            ax.set_title(f"Ground Truth: Serie {s_idx}, Bild {img_idx}", fontweight='bold')
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")
            
            # Hilfsgitter für Pixel-Bestimmung
            ax.grid(color='red', linestyle='--', linewidth=0.5, alpha=0.3)
            
            fig.colorbar(im, ax=ax, label="Normierte Intensität")
            
            # Speichern
            file_name = f"Reference_S{s_idx}_Img{img_idx}.png"
            fig.savefig(OUT_DIR / file_name, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Gespeichert: {file_name}")

    print(f"\nFertig! Alle Bilder liegen unter: {OUT_DIR}")

if __name__ == "__main__":
    main()