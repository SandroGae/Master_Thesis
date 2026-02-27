#!/usr/bin/env python3

import numpy as np
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =====================================================
# 1. KONFIGURATION
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
MOVIES_DIR = ROOT_DIR / "Plots" / "Unet" / "Movies_GroundTruth"

SERIES_LEN = 41
TOTAL_SERIES = 80 
NUM_LAST_SERIES = 8  # Wir wollen nur die letzten 8 (73-80)
FPS = 5

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def load_test_gt(h5_path: Path):
    """ Lädt nur die High-Count (GT) Daten """
    with h5py.File(h5_path, "r") as f:
        high_count = f["high_count/data"][:] # (H, W, N)
    
    # Umwandeln in (N, H, W)
    hc = np.moveaxis(high_count, -1, 0)
    return hc.astype(np.float32)

def normalize_slice_wise(series, scale=10000.0):
    """ Normalisiert jedes Bild der Serie einzeln auf seine Summe """
    # series shape: (41, 192, 240)
    series_clean = np.maximum(series, 0.0)
    # Summe pro Bild (axis 1 und 2)
    sums = np.sum(series_clean, axis=(1, 2), keepdims=True) + 1e-12
    return (series_clean / sums) * scale

def visual_norm(image):
    """ Skaliert für die Anzeige (0.5 bis 99.5 Perzentil) """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin < 1e-12:
        return np.zeros_like(image)
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

# =====================================================
# 3. MAIN
# =====================================================
def main():
    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Lade Ground Truth Daten von {H5_TEST_PATH}...")
    hc_all = load_test_gt(H5_TEST_PATH)
    
    # Die letzten 8 Serien berechnen (73 bis 80)
    # 0-basierte Indizes: 72 bis 79
    start_series_idx = TOTAL_SERIES - NUM_LAST_SERIES 
    
    for s_idx in range(start_series_idx, TOTAL_SERIES):
        series_num = s_idx + 1
        print(f"\nVerarbeite Serie {series_num} / {TOTAL_SERIES}...")
        
        # Extrahiere die 41 Bilder dieser Serie
        start_img = s_idx * SERIES_LEN
        end_img = (s_idx + 1) * SERIES_LEN
        series_data = hc_all[start_img:end_img]
        
        # Normalisierung (Beast V2 Standard)
        series_norm = normalize_slice_wise(series_data)
        
        frames = []
        for i in range(SERIES_LEN):
            img = series_norm[i]
            img_vis = visual_norm(img)
            
            # Einzel-Panel Plot
            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            ax.imshow(img_vis, cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"Ground Truth | Serie {series_num} | Bild {i+1}", fontsize=14, fontweight='bold')
            ax.axis("off")
            
            fig.tight_layout()
            fig.canvas.draw()
            
            # Canvas zu RGB konvertieren
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(frame)
            plt.close(fig)
            
        # Speichern
        out_path = MOVIES_DIR / f"GT_Only_Series_{series_num}.mp4"
        imageio.mimsave(str(out_path), frames, fps=FPS)
        print(f"✅ Video gespeichert: {out_path.name}")

    print("\nAlle Videos wurden erfolgreich erstellt.")

if __name__ == "__main__":
    main()