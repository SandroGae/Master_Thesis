# generate_movie_25d.py
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


# Konfiguration
MODEL_FILE_3stack ="unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras"
MODEL_FILE_5stack = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras"
MODEL_FILE_5stack_kernel_3x5 = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-103044_loss0.0523_val0.0587.keras"
MODEL_FILE_5stack_kernel_5x5_test = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-114917_loss0.0060_val0.0065.keras"


ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / MODEL_FILE_5stack_kernel_5x5_test
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"
MOVIES_DIR = ROOT_DIR / "Plots" / "Unet" / "Movies"

DEPTH = 5
SERIES_LEN = 41
SERIES_IDX = 12  # (1-basiert)
FPS = 3


def load_test_split(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]   # (H, W, N)
        high_count = f["high_count/data"][:] # (H, W, N)

    # (H, W, N) -> (N, H, W, 1)
    low_count = np.moveaxis(low_count, -1, 0)[..., np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[..., np.newaxis]

    return low_count.astype(np.float32), high_count.astype(np.float32)

def make_sliding_windows(X, y, series_len, depth):
    N = X.shape[0]
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    
    X_volumes = []
    y_volumes = []

    for i in range(n_series):
        start = i * series_len
        blockX = X[start:start + series_len]
        blockY = y[start:start + series_len]

        for start_idx in range(n_vols_per_series):
            # Slice (Depth, H, W, 1)
            X_volumes.append(blockX[start_idx:start_idx + depth])
            y_volumes.append(blockY[start_idx:start_idx + depth])

    return np.stack(X_volumes), np.stack(y_volumes), n_series, n_vols_per_series

def normalize_like_validation(volumes, scale=10000.0):
    """
    Normierung: ReLU -> SumNorm pro Slice -> Scale
    """
    volume = np.maximum(volumes, 0.0).astype(np.float32)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    volume = volume / sums
    volume = volume * scale
    return volume


# Visualisierung
def normalized_image(image):
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin < 1e-12: return image
    return (image - vmin) / (vmax - vmin)

def create_frames(X_seq_5d, Y_pred_25d, Y_true_5d, depth):
    """
    X_seq_5d:   (N, Depth, H, W, 1) -> Wir zeigen die mittlere Slice des Inputs
    Y_pred_25d: (N, H, W, 1)        -> Das Ergebnis des 2.5D Modells
    Y_true_5d:  (N, Depth, H, W, 1) -> Wir zeigen die mittlere Slice der Wahrheit
    """
    center_idx = depth // 2
    frames = []

    for i in range(X_seq_5d.shape[0]):
        inp_slice = X_seq_5d[i, center_idx, :, :, 0] # Input: Mittlere Slice aus dem Stack
        pred_slice = Y_pred_25d[i, :, :, 0] # Prediction: Das 2.5D Modell gibt direkt das Bild aus
        gt_slice = Y_true_5d[i, center_idx, :, :, 0] # Ground Truth: Mittlere Slice aus dem Stack

        # Normalisierung für Anzeige
        inp_norm  = normalized_image(inp_slice)
        pred_norm = normalized_image(pred_slice)
        gt_norm   = normalized_image(gt_slice)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        
        axes[0].imshow(inp_norm, cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title(f"Input (Middle Slice), Frame {i+1}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(pred_norm, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"2.5D Prediction, Frame {i+1}", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(gt_norm, cmap="gray_r", vmin=0, vmax=1)
        axes[2].set_title(f"Ground Truth, Frame {i+1}", fontsize=12)
        axes[2].axis("off")

        fig.tight_layout()
        fig.canvas.draw()
        
        # In Array wandeln
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        frames.append(frame)

    return frames


# Main
def main():
    s_idx_0 = SERIES_IDX - 1
    out_name = f"25D_Model_Series{SERIES_IDX}_Depth{DEPTH}_kernel_5x5_test.mp4"
    out_path = MOVIES_DIR / out_name

    model = models.load_model(MODEL_PATH, compile=False)
    X_test, y_test = load_test_split(H5_TEST_PATH)

    # Windows erstellen (ergibt 5D: N, Depth, H, W, 1)
    X_vols, y_vols, n_series, n_vols = make_sliding_windows(X_test, y_test, SERIES_LEN, DEPTH)
    
    # Serie auswählen
    start = s_idx_0 * n_vols
    end   = start + n_vols
    X_seq = X_vols[start:end] # (37, Depth, H, W, 1)
    Y_seq = y_vols[start:end]
    print(f"Nutze Serie {SERIES_IDX}, Volumen {start} bis {end}")

    # Normalisieren (wie im Training)
    X_seq_norm = normalize_like_validation(X_seq, scale=10000.0)
    # Ground Truth auch normalisieren für Anzeige
    Y_seq_norm = normalize_like_validation(Y_seq, scale=10000.0)

    # Letzte Dimension weg: (N, Depth, H, W)
    X_input_25d = np.squeeze(X_seq_norm, axis=-1)
    
    # Achsen tauschen: Depth (1) nach hinten (3)
    X_input_25d = np.transpose(X_input_25d, (0, 2, 3, 1))
    # Vorhersage
    Y_pred = model.predict(X_input_25d, batch_size=4, verbose=1) 
    # Output ist (N, H, W, 1)

     # Video erstellen
    frames = create_frames(X_seq_norm, Y_pred, Y_seq_norm, DEPTH)

    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print("Fertig.")

if __name__ == "__main__":
    main()