#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# =====================================================
# Konfiguration
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
DATA_DIR = ROOT_DIR / "original_data"
# Zielordner für das Video
IMAGES_ROOT_DIR = ROOT_DIR / "Plots" / "Unet" / "Movies"

H5_FILES = [
    DATA_DIR / "test_data.hdf5"
]

# Deine spezifische Auswahl an Serien (1-basiert)
SERIES_LIST = [5, 11, 12, 13, 15, 16, 21, 22, 29, 50]

SERIES_LEN = 41
FPS = 5  # Etwas schnellerer Flow für 410 Bilder

# =====================================================
# Hilfsfunktionen
# =====================================================
def load_high_count_only(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        high_count = f["high_count/data"][:] 
    high_count = np.moveaxis(high_count, -1, 0)
    high_count = high_count[..., np.newaxis]
    return high_count.astype(np.float32)

def normalize_slices_like_validation(slices, scale=10000.0):
    data = np.maximum(slices, 0.0).astype(np.float32)
    sums = np.sum(data, axis=(1, 2, 3), keepdims=True) + 1e-12
    return (data / sums) * scale

def normalized_image(image):
    # Geändert auf 2 und 98 Perzentil für den Clip
    vmin, vmax = np.percentile(image, [2, 98])
    if vmax - vmin < 1e-12:
        return np.zeros_like(image)
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def make_frame(image_norm, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(image_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape((h, w, 3))

    plt.close(fig)
    return frame

# =====================================================
# Hauptprozess
# =====================================================
def create_video_for_selection(h5_path: Path):
    print(f"\nGeneriere Video für {len(SERIES_LIST)} ausgewählte Serien aus: {h5_path.name}")
    split_name = h5_path.stem 

    # Daten laden
    gt_slices = load_high_count_only(h5_path)
    gt_slices_norm = normalize_slices_like_validation(gt_slices)

    frames = []
    
    # Progress Bar für die Gesamtzahl der Bilder (10 Serien * 41 Bilder)
    total_expected = len(SERIES_LIST) * SERIES_LEN
    pbar = tqdm(total=total_expected, desc="Rendering Frames")

    for s_idx in SERIES_LIST:
        # Umwandlung von 1-basiertem SERIES_LIST in 0-basierten Index für das Array
        base = (s_idx - 1) * SERIES_LEN
        
        for offset in range(SERIES_LEN):
            global_idx = base + offset
            
            # Sicherheitscheck für Indexgrenzen
            if global_idx >= gt_slices_norm.shape[0]:
                continue
                
            img = gt_slices_norm[global_idx, :, :, 0]
            img_disp = normalized_image(img)

            # Beschriftung 1-basiert
            title = f"{split_name} | Serie {s_idx} | Bild {offset + 1}"
            
            frame = make_frame(img_disp, title)
            frames.append(frame)
            pbar.update(1)

    pbar.close()

    if not frames:
        print("Fehler: Keine Bilder generiert.")
        return

    # Export
    IMAGES_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = IMAGES_ROOT_DIR / f"{split_name}_Selection_GT_2-98perc.mp4"

    print(f"Schreibe Video-Datei...")
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(str(out_file), fps=FPS, codec="libx264")
    print(f"\nErfolgreich erstellt: {out_file}")

def main():
    for h5_path in H5_FILES:
        if h5_path.exists():
            create_video_for_selection(h5_path)
        else:
            print(f"Datei nicht gefunden: {h5_path}")

if __name__ == "__main__":
    main()