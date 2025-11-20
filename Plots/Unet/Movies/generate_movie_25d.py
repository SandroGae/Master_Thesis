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

# =====================================================
# KONFIGURATION
# =====================================================
# Pfad zu deinem 2.5D Modell (.keras)
MODEL_FILE_3stack ="unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras"
MODEL_FILE_5stack = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras"


ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / MODEL_FILE_3stack
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"
MOVIES_DIR = ROOT_DIR / "Plots" / "Unet" / "Movies"

DEPTH = 3
SERIES_LEN = 41
SERIES_IDX = 50  # (1-basiert)
FPS = 3

# =====================================================
# Daten-Hilfsfunktionen
# =====================================================
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
    # Summe über (H,W,C) -> axis (2,3,4) bei Input (N, D, H, W, C)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    volume = volume / sums
    volume = volume * scale
    # WICHTIG: Falls du ohne Clip trainiert hast (V2), hier KEIN Clip.
    # Falls du doch clippen willst für Visualisierung, kannst du np.clip(..., 0, 1) nutzen.
    # Hier lassen wir es roh, wie im V2 Training.
    return volume

# =====================================================
# Visualisierung
# =====================================================
def normalized_image(image):
    # Percentile Scaling für hübsche Farben
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
        # 1. Input: Mittlere Slice aus dem Stack
        inp_slice = X_seq_5d[i, center_idx, :, :, 0]
        
        # 2. Prediction: Das 2.5D Modell gibt direkt das Bild aus
        pred_slice = Y_pred_25d[i, :, :, 0]
        
        # 3. Ground Truth: Mittlere Slice aus dem Stack
        gt_slice = Y_true_5d[i, center_idx, :, :, 0]

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

# =====================================================
# Main
# =====================================================
def main():
    s_idx_0 = SERIES_IDX - 1
    out_name = f"25D_Model_Series{SERIES_IDX}_Depth{DEPTH}.mp4"
    out_path = MOVIES_DIR / out_name

    print(f"--- 2.5D Video Generator ---")
    print(f"Modell: {MODEL_PATH}")
    print(f"Output: {out_path}")

    # 1. Modell laden
    # compile=False verhindert Fehler mit Custom Losses, wir brauchen nur predict()
    print("Lade Modell...")
    model = models.load_model(MODEL_PATH, compile=False)

    # 2. Daten laden
    print("Lade Testdaten...")
    X_test, y_test = load_test_split(H5_TEST_PATH)

    # 3. Windows erstellen (ergibt 5D: N, Depth, H, W, 1)
    print("Erstelle Windows...")
    X_vols, y_vols, n_series, n_vols = make_sliding_windows(X_test, y_test, SERIES_LEN, DEPTH)
    
    # 4. Richtige Serie auswählen
    start = s_idx_0 * n_vols
    end   = start + n_vols
    X_seq = X_vols[start:end] # (37, Depth, H, W, 1)
    Y_seq = y_vols[start:end]
    print(f"Nutze Serie {SERIES_IDX}, Volumen {start} bis {end}")

    # 5. Normalisieren (wie im Training)
    print("Normalisiere...")
    X_seq_norm = normalize_like_validation(X_seq, scale=10000.0)
    # Ground Truth auch normalisieren für Anzeige
    Y_seq_norm = normalize_like_validation(Y_seq, scale=10000.0)

    # =========================================================
    # DER ADAPTER-TRICK: 5D -> 4D (Channels)
    # =========================================================
    # Input ist (N, Depth, H, W, 1) -> Wir brauchen (N, H, W, Depth)
    print("Formatiere Input für 2.5D Modell um...")
    
    # 1. Letzte Dimension (1) weg: (N, Depth, H, W)
    X_input_25d = np.squeeze(X_seq_norm, axis=-1)
    
    # 2. Achsen tauschen: Depth (1) nach hinten (3)
    # (N, Depth, H, W) -> (N, H, W, Depth)
    # Permutation: 0->0, 1->3, 2->1, 3->2
    X_input_25d = np.transpose(X_input_25d, (0, 2, 3, 1))
    
    print(f"Input Shape für Modell: {X_input_25d.shape}")

    # 6. Vorhersage
    print("Berechne Prediction...")
    Y_pred = model.predict(X_input_25d, batch_size=4, verbose=1) 
    # Output ist (N, H, W, 1)

    # 7. Video erstellen
    print("Rendere Frames...")
    frames = create_frames(X_seq_norm, Y_pred, Y_seq_norm, DEPTH)

    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print("Fertig.")

if __name__ == "__main__":
    main()