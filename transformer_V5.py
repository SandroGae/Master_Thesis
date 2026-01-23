#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Helper-Skripte
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# -----------------------------
# Reproduzierbarkeit
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# -----------------------------
# Parameter
# -----------------------------
DEPTH = 5
SERIES_LEN = 41
EMBED_DIM = 96
WINDOW_SIZE = 8
BATCH_SIZE = 16 
INITIAL_LR = 5e-4
EPOCHS = 100

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
CKPT_FOLDER = "checkpoints_transformer"

# -----------------------------
# Hilfsfunktionen & Scheduler
# -----------------------------
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    return lr

def window_partition_func(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, window_size * window_size, C))

def window_reverse_func(windows, window_size, h, w):
    c = windows.shape[-1]
    x = tf.reshape(windows, (-1, h // window_size, w // window_size, window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, h, w, c))

def sw_block(x, dim, heads, window_size, shift=0):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    if shift > 0:
        x = tf.roll(x, shift=(-shift, -shift), axis=(1, 2))
    x_win = window_partition_func(x, window_size)
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(x_win, x_win)
    x = window_reverse_func(attn, window_size, h, w)
    if shift > 0:
        x = tf.roll(x, shift=(shift, shift), axis=(1, 2))
    x = layers.Add()([res, x])
    res2 = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(dim * 4, activation="gelu")(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([res2, x])

# -----------------------------
# Modellaufbau (Optimierte V4.1 Komplexität)
# -----------------------------
def build_srdtrans_v4_1(input_shape=(192, 240, 5), embed_dim=96):
    inputs = layers.Input(shape=input_shape, name="input")
    h, w, d = input_shape

    # 1. Temporal Attention Phase
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(inputs)
    attn_dim = 96 # Sync mit embed_dim für Effizienz
    xt = layers.Dense(attn_dim)(x)
    xt = layers.Reshape((h * w, d, attn_dim // d))(xt)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, d, attn_dim // d)))(xt)
    res_t = xt
    xt = layers.LayerNormalization(epsilon=1e-6)(xt)
    xt = layers.MultiHeadAttention(num_heads=4, key_dim=8)(xt, xt)
    xt = layers.Add()([res_t, xt])
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, h, w, attn_dim)))(xt)
    x = layers.Dense(embed_dim)(xt)

    # 2. Encoder Hierarchie
    # Ebene 1 (192x240) - 2 Blöcke
    x1 = sw_block(x, embed_dim, heads=4, window_size=8, shift=0)
    x1 = sw_block(x1, embed_dim, heads=4, window_size=8, shift=4)
    
    # Ebene 2 (96x120) - Erhöht auf 4 Blöcke (+2 gegenüber V4)
    p2 = layers.Conv2D(embed_dim * 2, kernel_size=3, strides=2, padding="same")(x1)
    x2 = sw_block(p2, embed_dim * 2, heads=8, window_size=8, shift=0)
    x2 = sw_block(x2, embed_dim * 2, heads=8, window_size=8, shift=4)
    x2 = sw_block(x2, embed_dim * 2, heads=8, window_size=8, shift=0)
    x2 = sw_block(x2, embed_dim * 2, heads=8, window_size=8, shift=4)
    
    # Ebene 3 (Bottleneck: 48x60) - Erhöht auf 4 Blöcke (+2 gegenüber V4)
    p3 = layers.Conv2D(embed_dim * 4, kernel_size=3, strides=2, padding="same")(x2)
    x3 = sw_block(p3, embed_dim * 4, heads=16, window_size=4, shift=0)
    x3 = sw_block(x3, embed_dim * 4, heads=16, window_size=4, shift=2)
    x3 = sw_block(x3, embed_dim * 4, heads=16, window_size=4, shift=0)
    x3 = sw_block(x3, embed_dim * 4, heads=16, window_size=4, shift=2)

    # 3. Decoder Hierarchie (Skip Connections)
    u2 = layers.Conv2DTranspose(embed_dim * 2, kernel_size=2, strides=2, padding="same")(x3)
    u2 = layers.Concatenate()([u2, x2]) 
    u2 = layers.Conv2D(embed_dim * 2, kernel_size=1)(u2) 
    u2 = sw_block(u2, embed_dim * 2, heads=8, window_size=8, shift=0)
    
    u1 = layers.Conv2DTranspose(embed_dim, kernel_size=2, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, x1]) 
    u1 = layers.Conv2D(embed_dim, kernel_size=1)(u1)
    u1 = sw_block(u1, embed_dim, heads=4, window_size=8, shift=0)

    # 4. Output
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(u1)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid", name="final_output")(x)

    return models.Model(inputs, outputs, name="srdtrans_V4_1_MediumHeavy")

# -----------------------------
# Daten-Pipeline & Loss
# -----------------------------
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]
        high_count = f["high_count/data"][:]
    low_count = np.moveaxis(low_count, -1, 0)
    high_count = np.moveaxis(high_count, -1, 0)
    low_count = low_count[:, :, :, np.newaxis]
    high_count = high_count[:, :, :, np.newaxis]
    return low_count, high_count

def make_sliding_windows(X, y, series_len=None, depth=None):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bX, bY = X[start:start+series_len], y[start:start+series_len]
        for start_idx in range(n_vols_per_series):
            X_v.append(bX[start_idx : start_idx + depth])
            y_v.append(bY[start_idx : start_idx + depth])
    return np.stack(X_v, axis=0), np.stack(y_v, axis=0)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x, y = x / sum_x, y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def prepare_transformer_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) 
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_val)

def mae_center(y_true, y_pred):
    y_t, y_p = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_t - y_p))

def mse_center(y_true, y_pred):
    y_t, y_p = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_t, y_p))

def psnr_center(y_true, y_pred):
    y_t, y_p = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_t, y_p), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_t, y_p = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))

# -----------------------------
# Main Run Loop
# -----------------------------
print("Lade Daten...")
X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

rng = np.random.default_rng(SEED)
idx_t = np.arange(len(X_train)); rng.shuffle(idx_t)
X_train, y_train = X_train[idx_t].astype(np.float32), y_train[idx_t].astype(np.float32)
idx_v = np.arange(len(X_val)); rng.shuffle(idx_v)
X_val, y_val = X_val[idx_v].astype(np.float32), y_val[idx_v].astype(np.float32)

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"transformer_V4_1__BS{BATCH_SIZE}__emb{EMBED_DIM}__{RUN_ID}"

# Verzeichnisse
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)
# CSV Pfad jetzt im TB Run Ordner
csv_path = TB_RUN_DIR / f"{RUN_NAME}.csv"

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_transformer_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_transformer_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

model = build_srdtrans_v4_1(embed_dim=EMBED_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

print(f"Training startet: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
    optimizer=optimizer, learning_rate=INITIAL_LR, input_shape=(192, 240, DEPTH),
    extra={"loss": "mae_ssim(alpha=0.6)", "model": "SRDTrans_V4_1_MediumHeavy"}
)

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print(f"Training beendet. Metriken unter {csv_path}")