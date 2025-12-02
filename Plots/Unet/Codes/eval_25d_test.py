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
DATA_ROOT = HOME / "data"
CODE_ROOT = HOME / "code" / "Master_Thesis" / "Plots" / "Unet"
H5_TEST_PATH = DATA_ROOT / "test_data_manipulated" / "test_every_second_image.hdf5"
MODEL_DIR = DATA_ROOT / "checkpoints_unet_3d_simple"
MODEL_FILE = "unet_25d_D5_VarStride1-24__20251201-093031_loss0.0690_val0.0594.keras"

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
    total_windows = n_series * n_per_series
    center_offset = depth // 2

    X_list, y_list = [], []

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
    model = models.load_model(model_path, compile=False)

    try:
        depth = model.input_shape[-1]
        if isinstance(depth, list):
            depth = depth[0]
    except:
        depth = 5

    X_test, y_test = create_windows(norm_x, norm_y, depth)

    print("Predict...")
    y_pred = model.predict(X_test, batch_size=32, verbose=1)

    metrics = calculate_metrics(y_test, y_pred)
    df = pd.DataFrame([{"Model": MODEL_FILE, "Depth": depth, **metrics}])

    out_csv = CODE_ROOT / "Analysis_ROI" / "Results_Normalized_10k.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"MAE {metrics['MAE']:.5f}, SSIM {metrics['SSIM']:.4f}")
    print("CSV:", out_csv)


if __name__ == "__main__":
    main()
