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

H5_TEST_PATH = DATA_DIR / "test_data.hdf5"

CLIP = False
SERIES_LEN = 41   # Laenge einer Serie
FPS = 3           # Bildwiederholrate des Videos

# Wir nehmen alle 41 Bilder pro Serie: 0,1,2,...,40
SLICE_OFFSETS = np.arange(0, SERIES_LEN, 1)

# Serienliste (1-basiert, wie du sie angegeben hast)
SERIES_LIST_1BASED = [12, 29, 50]


# =====================================================
# Daten-Hilfsfunktionen
# =====================================================
def load_low_and_high(h5_path: Path):
    """
    Laedt low_count und high_count aus HDF5-Datei und formatiert sie:
    (H, W, N) -> (N, H, W, 1)
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]   # (H, W, N)
        high_count = f["high_count/data"][:] # (H, W, N)

    low_count = np.moveaxis(low_count, -1, 0)    # (N, H, W)
    high_count = np.moveaxis(high_count, -1, 0)  # (N, H, W)

    low_count = low_count[..., np.newaxis]       # (N, H, W, 1)
    high_count = high_count[..., np.newaxis]     # (N, H, W, 1)

    return low_count.astype(np.float32), high_count.astype(np.float32)


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
# Visualisierung (LC links, HC rechts)
# =====================================================
def normalized_image(image):
    # Perzentile (0.5, 99.5)
    vmin, vmax = np.percentile(image, [0.1, 99.6])
    if vmax - vmin < 1e-12:
        return image  # quasi konstant
    return (image - vmin) / (vmax - vmin)


def make_frame(image_left_norm, image_right_norm, title):
    """
    Erzeugt einen einzelnen Frame (RGB-Array) mit Matplotlib,
    ohne auf die Platte zu schreiben.
    Links: LC, rechts: HC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    # Links: low-count
    axes[0].imshow(image_left_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[0].set_title("Low-count", fontsize=10)
    axes[0].axis("off")

    # Rechts: high-count (Ground Truth)
    axes[1].imshow(image_right_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[1].set_title("Ground truth", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape((h, w, 3))

    plt.close(fig)
    return frame


def create_video_for_selected_test_series(h5_path: Path):
    """
    Fuer test_data.hdf5:
      - low_count und high_count laden
      - beide wie Validation normieren
      - nur die Serien aus SERIES_LIST_1BASED verwenden
      - ALLE 41 Bilder pro gewaehlter Serie visualisieren
      - ein einziges Video mit LC links und HC rechts erzeugen
    Im Video: Titel "test_data - Serie X, Bild Y" pro Frame.
    """
    print(f"Verarbeite Test-Datei: {h5_path}")

    lc_slices, gt_slices = load_low_and_high(h5_path)   # (N, H, W, 1)
    lc_slices_norm = normalize_slices_like_validation(lc_slices, scale=10000.0)
    gt_slices_norm = normalize_slices_like_validation(gt_slices, scale=10000.0)

    N, H, W, C = gt_slices_norm.shape
    assert N % SERIES_LEN == 0, f"N={N} nicht durch SERIES_LEN={SERIES_LEN} teilbar"
    n_series = N // SERIES_LEN
    split_name = h5_path.stem  # "test_data"

    print(f"Anzahl Serien im Test-Set: {n_series}")

    # Serienliste von 1-basiert auf 0-basiert mappen und nur gueltige nehmen
    valid_series_indices = []
    for s in SERIES_LIST_1BASED:
        idx0 = s - 1
        if 0 <= idx0 < n_series:
            valid_series_indices.append(idx0)
        else:
            print(f"Achtung: Serie {s} (Index {idx0}) existiert nicht im Test-Set und wird ignoriert.")

    print(f"Verwende Serien (0-basiert): {valid_series_indices}")

    frames = []

    for series_idx in valid_series_indices:
        base = series_idx * SERIES_LEN

        for offset in SLICE_OFFSETS:
            global_idx = base + offset

            lc_img = lc_slices_norm[global_idx, :, :, 0]
            gt_img = gt_slices_norm[global_idx, :, :, 0]

            lc_disp = normalized_image(lc_img)
            gt_disp = normalized_image(gt_img)

            series_num = series_idx + 1  # wieder 1-basiert fuer Titel
            img_num = offset + 1

            title = f"{split_name} - Serie {series_num}, Bild {img_num}"
            frame = make_frame(lc_disp, gt_disp, title)
            frames.append(frame)

    out_dir = IMAGES_ROOT_DIR / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split_name}_LC_HC_selected_{FPS}fps.mp4"

    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(str(out_file), fps=FPS)
    print(f"Video gespeichert: {out_file}")


# =====================================================
# Hauptfunktion
# =====================================================
def main():
    print(f"ROOT_DIR:        {ROOT_DIR}")
    print(f"Images-Ordner:   {IMAGES_ROOT_DIR}")
    print(f"Test-HDF5:       {H5_TEST_PATH}")
    print(f"FPS:             {FPS}")
    print(f"Serien (1-basiert): {SERIES_LIST_1BASED}")

    create_video_for_selected_test_series(H5_TEST_PATH)

    print("Fertig.")


if __name__ == "__main__":
    main()
