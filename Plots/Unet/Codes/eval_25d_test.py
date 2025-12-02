#!/usr/bin/env python3

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

HOME = Path("/home/sgaell")
DATA_ROOT = HOME / "data" / "original_data" / "test_data.hdf5"
CODE_ROOT = HOME / "code" / "Master_Thesis" / "Plots" / "Unet"
H5_TEST_PATH = DATA_ROOT / "test_data_manipulated" / "test_every_third_image.hdf5"
# H5_TEST_PATH = DATA_ROOT / "test_data_manipulated" / "test_every_third_image.hdf5"
MODEL_DIR = DATA_ROOT / "checkpoints_unet_3d_simple"
MODEL_FILE = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras"

SERIES_LEN = 41
NORM_SCALE = 10000.0


def normalize_fixed(volume: np.ndarray) -> np.ndarray:
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume, axis=(1, 2, 3), keepdims=True) + 1e-12
    return (volume / sums) * NORM_SCALE


def load_test_data(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        low = f["low_count/data"][:]
        high = f["high_count/data"][:]
    low = np.moveaxis(low, -1, 0)[..., np.newaxis].astype(np.float32)
    high = np.moveaxis(high, -1, 0)[..., np.newaxis].astype(np.float32)
    return low, high


def create_windows(X: np.ndarray, y: np.ndarray, depth: int):
    N = X.shape[0]
    n_series = N // SERIES_LEN
    n_per_series = SERIES_LEN - depth + 1
    # center_offset: Welches Bild ist das Ziel? (Meistens das mittlere)
    center_offset = depth // 2

    X_list, y_list = [], []

    # Iteriere über jede Serie (z.B. jeden Patienten/Scan)
    for s in range(n_series):
        # Startindex der aktuellen Serie
        start_index = s * SERIES_LEN
        
        # Iteriere innerhalb der Serie (Sliding Window)
        for i in range(n_per_series):
            # Indizes berechnen
            idx_start = start_index + i
            idx_end = idx_start + depth
            
            # Input-Volume ausschneiden (Dimension: Depth, H, W, C)
            x_window = X[idx_start:idx_end]

            if x_window.shape[-1] == 1:
                x_window = np.transpose(x_window, (1, 2, 0, 3)) # -> (H, W, Depth, 1)
                x_window = np.squeeze(x_window, axis=-1)        # -> (H, W, Depth)
            
            # Target Image (das mittlere Bild im Fenster)
            y_target = y[idx_start + center_offset]
            
            X_list.append(x_window)
            y_list.append(y_target)

    return np.array(X_list), np.array(y_list)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_t = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_p = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    mae = tf.reduce_mean(tf.abs(y_t - y_p)).numpy()
    mse = tf.reduce_mean(tf.square(y_t - y_p)).numpy()
    psnr = tf.reduce_mean(tf.image.psnr(y_t, y_p, max_val=1.0)).numpy()
    ssim = tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0)).numpy()
    try:
        ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(y_t, y_p, max_val=1.0)).numpy()
    except Exception:
        ms_ssim = np.nan
    return {"MAE": mae, "MSE": mse, "PSNR": psnr, "SSIM": ssim, "MS-SSIM": ms_ssim}


def main():
    raw_x, raw_y = load_test_data(H5_TEST_PATH)
    norm_x = normalize_fixed(raw_x)
    norm_y = normalize_fixed(raw_y)

    model_path = MODEL_DIR / MODEL_FILE
    print(f"Loading model: {model_path.name} ...")
    model = models.load_model(model_path, compile=False)

    try:
        depth = model.input_shape[-1]
        if isinstance(depth, list):
            depth = depth[0]
    except:
        depth = 5

    # Denke dran, hier die korrigierte create_windows Funktion zu nutzen (siehe oben)
    print("Creating windows...")
    X_test, y_test = create_windows(norm_x, norm_y, depth)

    print(f"Predicting on {len(X_test)} samples...")
    y_pred = model.predict(X_test, batch_size=32, verbose=1)

    metrics = calculate_metrics(y_test, y_pred)

    # --- Einfache, saubere Ausgabe in der Konsole ---
    print("\n" + "="*40)
    print(f"  EVALUATION RESULTS")
    print("="*40)
    print(f"Model:   {MODEL_FILE}")
    print(f"Samples: {len(X_test)}")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:<10}: {value:.6f}") # 6 Nachkommastellen für Details
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
