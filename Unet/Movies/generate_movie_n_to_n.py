# generate_movie_n_to_n.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# Pfade
# =====================================================
FILE_CLIP = "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-084215_loss0.0496_val0.0531.keras"
FILE_NO_CLIP = "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006_loss0.0489_val0.0524.keras"

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / FILE_NO_CLIP
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
MOVIES_DIR = ROOT_DIR / "Plots" / "Unet" / "Movies"

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


def normalize_like_validation(volumes, scale=10000.0, do_clip = CLIP):
    """
    Normierung wie im Validation-Set:
    - clip >= 0
    - pro Slice durch Summe über (H,W,C) teilen
    - mit 'scale' multiplizieren
    - KEIN FINALE CLIP!!!
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
# Frame-Erzeugung
# =====================================================
def normalized_image(image):
    vmin, vmax = np.percentile(image, [0.5, 99.5])  # gleiche Logik wie im Bildscript
    if vmax - vmin < 1e-12:
        return image  # Bild ist quasi konstant
    return (image - vmin) / (vmax - vmin)


def create_frames_from_sequence(X_seq, Y_pred, Y_true, depth):
    center_idx = depth // 2
    frames = []

    for i in range(X_seq.shape[0]):
        inp_slice  = X_seq[i, center_idx, :, :, 0]
        pred_slice = Y_pred[i, center_idx, :, :, 0]
        gt_slice   = Y_true[i, center_idx, :, :, 0]

        inp_norm  = normalized_image(inp_slice)
        pred_norm = normalized_image(pred_slice)
        gt_norm   = normalized_image(gt_slice)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)  # GRÖSSERES VIDEO

        axes[0].imshow(inp_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Input, Vol {i}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(pred_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Prediction, Vol {i}", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(gt_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Ground Truth, Vol {i}", fontsize=12)
        axes[2].axis("off")

        fig.tight_layout()

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

        plt.close(fig)

        frames.append(frame)

    return frames



# =====================================================
# Hauptfunktion
# =====================================================
def main():
    series_idx = 50   # Serie 1 basiert
    fps = 3           # FPS einstellen

    series_idx -= 1

    out_name = f"{MODEL_PATH.stem}_series{series_idx + 1}.mp4"
    out_path = MOVIES_DIR / out_name

    print(f"Modell-Datei: {MODEL_PATH}")
    print(f"Test-HDF5:    {H5_TEST_PATH}")
    print(f"Serie:        {series_idx}")
    print(f"FPS:          {fps}")
    print(f"Output-Video: {out_path}")

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
    print(f"Nutze Volumen [{start_idx}:{end_idx}) von Serie {series_idx + 1}")

    X_seq = X_vols[start_idx:end_idx]
    Y_seq = y_vols[start_idx:end_idx]


    print("Normalisiere Volumen...")
    X_seq_norm = normalize_like_validation(X_seq, scale=10000.0)
    Y_seq_norm = normalize_like_validation(Y_seq, scale=10000.0)


    print("Berechne Predictions...")
    Y_pred = model.predict(X_seq_norm, batch_size=1, verbose=1)

    print("Erzeuge Frames...")
    frames = create_frames_from_sequence(X_seq_norm, Y_pred, Y_seq_norm, DEPTH)

    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Schreibe Video nach {out_path} mit FPS={fps} ...")
    imageio.mimsave(str(out_path), frames, fps=fps)
    print("Fertig.")



if __name__ == "__main__":
    main()