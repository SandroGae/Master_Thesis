#!/usr/bin/env python3

import os
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# GPU Memory Growth aktivieren
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# --- Konfiguration ---
HOME = Path("/home/sgaell")
DATA_ROOT = HOME / "data"
CODE_ROOT = HOME / "code" / "Master_Thesis" / "Plots" / "Unet"

# Pfade anpassen falls nötig
H5_TEST_PATH = DATA_ROOT / "original_data" / "test_data.hdf5"
MODEL_DIR = DATA_ROOT / "checkpoints_unet_3d_simple"
MODEL_FILE = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251202-121350_loss0.0527_val0.0587.keras"

SERIES_LEN = 41
NORM_SCALE = 10000.0


def normalize_fixed(volume: np.ndarray) -> np.ndarray:
    """Normalisiert auf Summe NORM_SCALE."""
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume, axis=(1, 2, 3), keepdims=True) + 1e-12
    return (volume / sums) * NORM_SCALE


def load_test_data(h5_path: Path):
    """Lädt Testdaten."""
    with h5py.File(h5_path, "r") as f:
        low = f["low_count/data"][:]
        high = f["high_count/data"][:]
    # Dimensionen anpassen: (N, H, W, 1)
    low = np.moveaxis(low, -1, 0)[..., np.newaxis].astype(np.float32)
    high = np.moveaxis(high, -1, 0)[..., np.newaxis].astype(np.float32)
    return low, high


def create_windows(X: np.ndarray, y: np.ndarray, depth: int):
    """Erstellt 2.5D Input Windows."""
    N = X.shape[0]
    n_series = N // SERIES_LEN
    n_per_series = SERIES_LEN - depth + 1
    center_offset = depth // 2

    # Vorallokation für bessere Performance (optional, hier Listen-Ansatz wie gehabt)
    X_list, y_list = [], []

    for s in range(n_series):
        start_index = s * SERIES_LEN
        for i in range(n_per_series):
            idx_start = start_index + i
            idx_end = idx_start + depth
            
            # Input Volume
            x_window = X[idx_start:idx_end] # (D, H, W, 1)

            # Transpose für Modell (H, W, D)
            if x_window.shape[-1] == 1:
                x_window = np.transpose(x_window, (1, 2, 0, 3)) 
                x_window = np.squeeze(x_window, axis=-1) 
            
            # Target (Center Frame)
            y_target = y[idx_start + center_offset]
            
            X_list.append(x_window)
            y_list.append(y_target)

    return np.array(X_list), np.array(y_list)


def calculate_metrics_dual(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Berechnet Metriken einmal MIT Clipping (0..1) und einmal OHNE (Raw).
    """
    y_t = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_p = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # --- 1. Variante: CLIPPED (Vergleichbar mit Training) ---
    # Alles über 1.0 wird auf 1.0 gesetzt.
    y_t_clip = tf.clip_by_value(y_t, 0.0, 1.0)
    y_p_clip = tf.clip_by_value(y_p, 0.0, 1.0)

    mae_c = tf.reduce_mean(tf.abs(y_t_clip - y_p_clip)).numpy()
    mse_c = tf.reduce_mean(tf.square(y_t_clip - y_p_clip)).numpy()
    psnr_c = tf.reduce_mean(tf.image.psnr(y_t_clip, y_p_clip, max_val=1.0)).numpy()
    ssim_c = tf.reduce_mean(tf.image.ssim(y_t_clip, y_p_clip, max_val=1.0)).numpy()
    
    # --- 2. Variante: RAW (Unclipped) ---
    # Hier nehmen wir die tatsächlichen Werte (können > 1.0 sein).
    # PSNR/SSIM brauchen einen max_val. Da die Daten auf 10000 normiert sind, 
    # kann ein Pixel theoretisch sehr hoch sein. 
    # Sinnvollerweise nehmen wir hier entweder das Daten-Maximum oder bleiben formal bei 1.0,
    # was aber zu schlechten SSIM Werten führt wenn Werte > 1.0 sind.
    # Für einen fairen Vergleich der Fehler nutzen wir hier einfach die Differenz.
    
    mae_r = tf.reduce_mean(tf.abs(y_t - y_p)).numpy()
    mse_r = tf.reduce_mean(tf.square(y_t - y_p)).numpy()
    
    # Für PSNR/SSIM auf Raw Daten müssen wir entscheiden, was "Peak" ist.
    # Da das Netz Sigmoid hat (0..1), ist y_p eh max 1.0. 
    # y_t kann aber > 1 sein. Wir nutzen max_val=max(y_t) für "fairen" PSNR auf Raw Daten?
    # Oder wir lassen max_val=1.0, was üblich ist, wenn der Wertebereich formal 0..1 sein soll.
    # Hier: max_val=1.0 (Standard), zeigt wie stark die "Hot Spots" bestraft werden.
    psnr_r = tf.reduce_mean(tf.image.psnr(y_t, y_p, max_val=1.0)).numpy()
    ssim_r = tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0)).numpy()

    return {
        "clipped": {"MAE": mae_c, "MSE": mse_c, "PSNR": psnr_c, "SSIM": ssim_c},
        "raw":     {"MAE": mae_r, "MSE": mse_r, "PSNR": psnr_r, "SSIM": ssim_r}
    }


def main():
    # 1. Daten laden
    print(f"Lade Daten von: {H5_TEST_PATH}")
    raw_x, raw_y = load_test_data(H5_TEST_PATH)
    
    # 2. Normalisieren
    print(f"Normalisiere (Scale={NORM_SCALE})...")
    norm_x = normalize_fixed(raw_x)
    norm_y = normalize_fixed(raw_y)

    # 3. Modell laden
    model_path = MODEL_DIR / MODEL_FILE
    print(f"Lade Modell: {model_path.name} ...")
    model = models.load_model(model_path, compile=False)

    # Tiefe ermitteln
    try:
        depth = model.input_shape[-1]
        if isinstance(depth, list): depth = depth[0]
    except:
        depth = 5

    # 4. Windows erstellen
    print(f"Erstelle Windows (Depth={depth})...")
    X_test, y_test = create_windows(norm_x, norm_y, depth)

    # 5. Prediction
    print(f"Starte Prediction auf {len(X_test)} Samples...")
    y_pred = model.predict(X_test, batch_size=32, verbose=1)

    # 6. Metriken berechnen (Beide Varianten)
    print("Berechne Metriken...")
    results = calculate_metrics_dual(y_test, y_pred)

    # 7. Ausgabe
    print("\n" + "="*65)
    print(f"  EVALUATION RESULTS: {MODEL_FILE}")
    print("="*65)
    print(f"{'Metric':<10} | {'CLIPPED (0..1)':<20} | {'RAW / UNCLIPPED':<20}")
    print("-" * 65)
    
    metrics_list = ["MAE", "MSE", "PSNR", "SSIM"]
    
    for m in metrics_list:
        val_c = results["clipped"][m]
        val_r = results["raw"][m]
        
        # Formatierung
        print(f"{m:<10} | {val_c:<20.6f} | {val_r:<20.6f}")
        
    print("-" * 65)
    print("NOTE: 'Clipped' entspricht der Trainings-Logik (ignore > 1.0).")
    print("      'Raw' bestraft Abweichungen bei Werten > 1.0 (Hot Spots).")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()