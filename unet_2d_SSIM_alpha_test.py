#!/usr/bin/env python3
# unet_2d_SSIM_alpha_test.py

import os, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Reproduzierbarkeit & (optionale) GPU-Memory-Growth ---
SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)
try:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# --- Pfade ---
FILES = {
    "training":   "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}
CSV_OUT = Path.home() / "data" / "alpha_sweep_unet_2d_SSIM.csv"

# --- Daten I/O ---
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_count  = f["low_count/data"][:]   # (H,W,N)
        high_count = f["high_count/data"][:]  # (H,W,N)
    low_count  = np.moveaxis(low_count,  -1, 0)[:, :, :, np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[:, :, :, np.newaxis]
    return low_count.astype(np.float32), high_count.astype(np.float32)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X)); rng.shuffle(idx)
    return X[idx], y[idx]

def augment_and_normalize_2d(scale_min: float, scale_max: float, p: float = 0.5):
    def map_picture(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [1]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [1]), lambda: y)
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, [0,1,2], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, [0,1,2], keepdims=True) + 1e-12
        x = x / sum_x; y = y / sum_y
        scale = tf.random.uniform([1,1,1], scale_min, scale_max, dtype=tf.float32)
        x = tf.clip_by_value(x * scale, 0.0, 1.0)
        y = tf.clip_by_value(y * scale, 0.0, 1.0)
        return x, y
    return map_picture

# --- Modell ---
POOL_HW = (2, 2)

def conv_block_2d(x, filters, kernel_size=(3,3), padding="same"):
    ki = "he_normal"
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x); x = layers.ReLU()(x)
    return x

def unet_2d(input_shape=(192,240,1), base_filters=16, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, base_filters)              ; p1 = layers.MaxPooling2D(POOL_HW)(c1)
    c2 = conv_block_2d(p1, base_filters*2)                ; p2 = layers.MaxPooling2D(POOL_HW)(c2)
    c3 = conv_block_2d(p2, base_filters*4)                ; p3 = layers.MaxPooling2D(POOL_HW)(c3)
    c4 = conv_block_2d(p3, base_filters*8)                ; p4 = layers.MaxPooling2D(POOL_HW)(c4)
    bn = conv_block_2d(p4, base_filters*16)
    u4 = layers.Conv2DTranspose(base_filters*8, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])                   ; c5 = conv_block_2d(u4, base_filters*8)
    u3 = layers.Conv2DTranspose(base_filters*4, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])                   ; c6 = conv_block_2d(u3, base_filters*4)
    u2 = layers.Conv2DTranspose(base_filters*2, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])                   ; c7 = conv_block_2d(u2, base_filters*2)
    u1 = layers.Conv2DTranspose(base_filters, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])                   ; c8 = conv_block_2d(u1, base_filters)
    out = layers.Conv2D(1, (1,1), activation=output_activation, kernel_initializer="he_normal", use_bias=True, name="output")(c8)
    return models.Model(inputs, out, name="unet_2d_simple_relu_sigmoid")

# --- Loss & Metriken ---
def combined_mae_ssim(y_true, y_pred, alpha=0.7):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * (10*mae) + alpha * (1.0 - ssim)

def psnr_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# --- Einziger CSV-Logger (append) mit Alpha-Feld ---
class OneCSVPerAllRuns(tf.keras.callbacks.Callback):
    header_written = False
    def __init__(self, csv_path, run_name, alpha):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.run_name = run_name
        self.alpha = alpha
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp","run_name","alpha","epoch",
                    "loss","val_loss",
                    "mae","mse","psnr_metric","ssim_metric",
                    "val_mae","val_mse","val_psnr_metric","val_ssim_metric",
                ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = [
            datetime.now().isoformat(timespec="seconds"),
            self.run_name,
            f"{self.alpha:.2f}",
            epoch + 1,
            logs.get("loss"), logs.get("val_loss"),
            logs.get("mae"),  logs.get("mse"),  logs.get("psnr_metric"),  logs.get("ssim_metric"),
            logs.get("val_mae"), logs.get("val_mse"), logs.get("val_psnr_metric"), logs.get("val_ssim_metric"),
        ]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)

# --- Daten laden & Pipelines bauen (wie bei dir) ---
print("Lade Daten...")
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_2d(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .map(augment_and_normalize_2d(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

# --- Alpha-Sweep: 0.00 bis 1.00 in 0.05-Schritten, jeweils 20 Epochen ---
ALPHAS = [i/100 for i in range(0, 101, 5)]  # 0.00, 0.05, ..., 1.00
BASE_NAME = "unet_2d_SSIM"
EPOCHS = 20

for alpha in ALPHAS:
    tf.keras.backend.clear_session()

    RUN_ID   = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_NAME = f"{BASE_NAME}__seed{SEED}__bf{16}__alpha{alpha:.2f}__{RUN_ID}"

    model = unet_2d(input_shape=(192,240,1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
    # alpha sauber binden!
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred, a=alpha: combined_mae_ssim(y_true, y_pred, alpha=a),
        metrics=['mae','mse', psnr_metric, ssim_metric],
    )

    csv_cb = OneCSVPerAllRuns(CSV_OUT, RUN_NAME, alpha)

    print(f"\n=== Starte Run: {RUN_NAME} ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[csv_cb],
        verbose=2
    )

print(f"\nFertig. Alle Ergebnisse stehen in: {CSV_OUT}")
