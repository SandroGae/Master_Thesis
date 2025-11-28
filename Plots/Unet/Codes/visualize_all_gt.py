#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

# =====================================================
# Pfade
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
DATA_DIR = ROOT_DIR / "original_data"
IMAGES_ROOT_DIR = ROOT_DIR / "Plots" / "Unet" / "Images"

H5_FILES = [
    DATA_DIR / "training_data.hdf5",
    DATA_DIR / "validation_data.hdf5",
    DATA_DIR / "test_data.hdf5",
]

CLIP = False
SERIES_LEN = 41   # Laenge einer Serie
FPS = 3           # Bildwiederholrate der Videos

# wir wollen jetzt JEDE Slice visualisieren: 0,1,2,...,40
SLICE_OFFSETS = np.arange(0, SERIES_LEN, 1)


# =====================================================
# Daten-Hilfsfunktionen
# =====================================================
def load_high_count_only(h5_path: Path):
    """
    Laedt NUR die high_count-Daten aus einer HDF5-Datei und formatiert sie:
    (H, W, N) -> (N, H, W, 1)
    """
    with h5py.File(h5_path, "r") as f:
        high_count = f["high_count/data"][:]  # (H, W, N)

    high_count = np.moveaxis(high_count, -1, 0)   # (N, H, W)
    high_count = high_count[..., np.newaxis]      # (N, H, W, 1)

    return high_count.astype(np.float32)


def normalize_slices_like_validation(slices, scale=10000.0, do_clip=CLIP):
    """
    Normierung analog zur Validation-Normierung, aber ohne Volumen:
    - clip >= 0
    - pro Slice durch Summe ueber (H,W,C) teilen
    - mit 'scale' multiplizieren
    slices: (N, H, W, 1)
    """
    data = np.maximum(slices, 0.0).astype(np.float32)
    sums = np.sum(data, axis=(1, 2, 3), keepdims=True) + 1e-12
    data = data / sums
    data = data * scale
    if do_clip:
        data = np.clip(data, 0.0, 1.0)
    return data


# =====================================================
# Visualisierung (nur fuer Ground Truth)
# =====================================================
def normalized_image(image):
    # Gewuenschte Perzentile (2, 92)
    vmin, vmax = np.percentile(image, [2, 92])
    if vmax - vmin < 1e-12:
        return image  # quasi konstant
    return (image - vmin) / (vmax - vmin)


def make_frame(image_norm, title):
    """
    Erzeugt einen einzelnen Frame (RGB-Array) mit Matplotlib,
    ohne auf die Platte zu schreiben.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)
    ax.imshow(image_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape((h, w, 3))

    plt.close(fig)
    return frame


def create_video_for_file(h5_path: Path):
    """
    Fuer eine HDF5-Datei:
      - high_count laden
      - wie Validation normieren
      - ALLE Bilder (Slices) visualisieren
      - pro Datei genau ein Video mit nur Ground Truth erzeugen
    Im Video: Titel "<split_name> - Serie X, Bild Y" pro Frame.
    """
    print(f"Verarbeite Datei: {h5_path}")
    split_name = h5_path.stem  # z.B. "training_data"

    gt_slices = load_high_count_only(h5_path)           # (N, H, W, 1)
    gt_slices_norm = normalize_slices_like_validation(gt_slices, scale=10000.0)

    N, H, W, C = gt_slices_norm.shape
    assert N % SERIES_LEN == 0, f"N={N} nicht durch SERIES_LEN={SERIES_LEN} teilbar"
    n_series = N // SERIES_LEN

    frames = []

    for series_idx in range(n_series):
        base = series_idx * SERIES_LEN

        for offset in SLICE_OFFSETS:
            global_idx = base + offset

            img = gt_slices_norm[global_idx, :, :, 0]
            img_disp = normalized_image(img)

            series_num = series_idx + 1
            img_num = offset + 1

            title = f"{split_name} - Serie {series_num}, Bild {img_num}"
            frame = make_frame(img_disp, title)
            frames.append(frame)

    out_dir = IMAGES_ROOT_DIR / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split_name}_groundtruth_{FPS}fps.mp4"

    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(str(out_file), fps=FPS)
    print(f"Video gespeichert: {out_file}")


# =====================================================
# Hauptfunktion
# =====================================================
def main():
    print(f"ROOT_DIR:        {ROOT_DIR}")
    print(f"Images-Ordner:   {IMAGES_ROOT_DIR}")
    print(f"HDF5-Dateien:    {H5_FILES}")
    print(f"FPS:             {FPS}")

    for h5_path in H5_FILES:
        create_video_for_file(h5_path)

    print("Fertig.")


if __name__ == "__main__":
    main()
