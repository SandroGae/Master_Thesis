#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# Pfade
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / "unet_3d_SSIM_middle__seed42__bf64__D3__lossMAE_SSIM__20251112-231304_loss0.0481_val0.0518.keras"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
IMAGES_ROOT_DIR = ROOT_DIR / "Plots" / "Unet" / "Images"

CLIP = False
DEPTH = 3
SERIES_LEN = 41
N_VOLS_PER_SERIES = SERIES_LEN - DEPTH + 1  # = 37


# =====================================================
# Daten-Hilfsfunktionen
# =====================================================
def load_test_split(h5_path: Path):
    """
    Laedt Testdaten aus HDF5-Datei und formatiert sie:
    (H, W, N) -> (N, H, W, 1)
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]   # (H, W, N)
        high_count = f["high_count/data"][:] # (H, W, N)

    low_count = np.moveaxis(low_count, -1, 0)   # (N, H, W)
    high_count = np.moveaxis(high_count, -1, 0) # (N, H, W)

    low_count = low_count[..., np.newaxis]      # (N, H, W, 1)
    high_count = high_count[..., np.newaxis]    # (N, H, W, 1)

    return low_count.astype(np.float32), high_count.astype(np.float32)


def make_sliding_windows(X, y, series_len, depth):
    """
    X, y: (N, H, W, 1), N = num_series * series_len
    Output:
        X_vols, y_vols: (N_vols_total, D, H, W, 1)
        n_series: Anzahl Serien
        n_vols_per_series: Volumen pro Serie (= series_len - depth + 1)
    """
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1

    X_volumes = []
    y_volumes = []

    for i in range(n_series):
        start = i * series_len
        blockX = X[start:start + series_len]
        blockY = y[start:start + series_len]

        for start_idx in range(n_vols_per_series):
            X_volumes.append(blockX[start_idx:start_idx + depth])
            y_volumes.append(blockY[start_idx:start_idx + depth])

    X_volumes = np.stack(X_volumes, axis=0)
    y_volumes = np.stack(y_volumes, axis=0)
    return X_volumes, y_volumes, n_series, n_vols_per_series


def normalize_like_validation(volumes, scale=10000.0, do_clip=CLIP):
    """
    Normierung wie im Validation-Set:
    - clip >= 0
    - pro Slice durch Summe ueber (H,W,C) teilen
    - mit 'scale' multiplizieren
    volumes: (N, D, H, W, 1)
    """
    volume = np.maximum(volumes, 0.0).astype(np.float32)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    volume = volume / sums
    volume = volume * scale
    if do_clip:
        volume = np.clip(volume, 0.0, 1.0)
    return volume


# =====================================================
# Visualisierung
# =====================================================
def normalized_image(image):
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin < 1e-12:
        return image  # quasi konstant
    return (image - vmin) / (vmax - vmin)


def save_single_image(img_norm, title, out_file: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)
    ax.imshow(img_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_file}")


def save_series_into_subfolders(X_seq, Y_pred, Y_true, depth, root_dir: Path, series_idx: int):
    """
    Speichert für alle Volumen:
      - Input (train)  -> root_dir / seriesXX / input
      - Prediction     -> root_dir / seriesXX / prediction
      - Ground Truth   -> root_dir / seriesXX / ground_truth
    jeweils als einzelne PNGs (nur Zentralslice).
    """
    center_idx = depth // 2   # bei DEPTH=3 -> 1

    series_dir = root_dir / f"series{series_idx}"
    inp_dir = series_dir / "input"
    pred_dir = series_dir / "prediction"
    gt_dir = series_dir / "ground_truth"

    for i in range(X_seq.shape[0]):
        # Input und Ground Truth: haben DEPTH in Achse 1
        inp_slice  = X_seq[i, center_idx, :, :, 0]
        gt_slice   = Y_true[i, center_idx, :, :, 0]

        # Prediction: hat nur 1 in der Depth-Achse -> immer Index 0
        pred_slice = Y_pred[i, 0, :, :, 0]

        inp_norm  = normalized_image(inp_slice)
        pred_norm = normalized_image(pred_slice)
        gt_norm   = normalized_image(gt_slice)

        # Dateinamen
        inp_file  = inp_dir  / f"{MODEL_PATH.stem}_series{series_idx}_vol{i:03d}.png"
        pred_file = pred_dir / f"{MODEL_PATH.stem}_series{series_idx}_vol{i:03d}.png"
        gt_file   = gt_dir   / f"{MODEL_PATH.stem}_series{series_idx}_vol{i:03d}.png"

        # speichern
        save_single_image(inp_norm,  f"Input, Vol {i}",      inp_file)
        save_single_image(pred_norm, f"Prediction, Vol {i}", pred_file)
        save_single_image(gt_norm,   f"Ground Truth, Vol {i}", gt_file)


# =====================================================
# Hauptfunktion
# =====================================================
def main():
    series_idx = 11   # Serie wählen 0 basiert (11-->Serie 12)

    print(f"Modell-Datei:  {MODEL_PATH}")
    print(f"Test-HDF5:     {H5_TEST_PATH}")
    print(f"Serie:         {series_idx}")
    print(f"Output-Root:   {IMAGES_ROOT_DIR}")

    print("Lade Modell...")
    model = models.load_model(MODEL_PATH, compile=False)

    print("Lade Testdaten...")
    X_test, y_test = load_test_split(H5_TEST_PATH)

    print("Erzeuge Volumen per Sliding Window...")
    X_vols, y_vols, n_series, n_vols_per_series = make_sliding_windows(
        X_test, y_test, SERIES_LEN, DEPTH
    )

    assert n_vols_per_series == N_VOLS_PER_SERIES
    assert 0 <= series_idx < n_series

    start_idx = series_idx * n_vols_per_series
    end_idx = start_idx + n_vols_per_series
    print(f"Nutze Volumen [{start_idx}:{end_idx}) von Serie {series_idx}")

    X_seq = X_vols[start_idx:end_idx]
    Y_seq = y_vols[start_idx:end_idx]

    print("Normalisiere Volumen...")
    X_seq_norm = normalize_like_validation(X_seq, scale=10000.0)
    Y_seq_norm = normalize_like_validation(Y_seq, scale=10000.0)

    print("Berechne Predictions...")
    Y_pred = model.predict(X_seq_norm, batch_size=1, verbose=1)

    print("Speichere Bilder in getrennten Unterordnern (input/prediction/ground_truth)...")
    save_series_into_subfolders(X_seq_norm, Y_pred, Y_seq_norm, DEPTH, IMAGES_ROOT_DIR, series_idx)

    print("Fertig.")


if __name__ == "__main__":
    main()
