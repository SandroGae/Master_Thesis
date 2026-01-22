#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold

from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# Reproduzierbarkeit
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Parameters
DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64
BATCH_SIZE = 16 # Changed from 8 --> 16
EPOCHS = 100
ALPHA_FIXED = 0.06  # Dein Zielwert
N_FOLDS = 5

# --- ARCHITEKTUR ---

def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block_2d(inputs, base_filters)          ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2)          ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4)          ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8)          ; p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bn = conv_block_2d(p4, base_filters * 16)

    # Decoder
    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])               ; c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])               ; c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])               ; c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])               ; c8 = conv_block_2d(u1, base_filters)

    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8)
    
    return models.Model(inputs, out, name="unet_25d_stacked")

# --- DATEN-PIPELINE ---

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
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1

    X_volumes = []
    y_volumes = []

    for i in range(0, n_series, 1):
        start = i * series_len
        blockX = X[start:start+series_len]
        blockY = y[start:start+series_len]

        for start_idx in range(0, n_vols_per_series, 1):
            X_volumes.append(blockX[start_idx : start_idx + depth])
            y_volumes.append(blockY[start_idx : start_idx + depth])

    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)

def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    def map_volume(x, y):
        flip = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y
        scale = tf.random.uniform([], minval=scale_min, maxval=scale_max, dtype=tf.float32)
        return x * scale, y * scale
    return map_volume

def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 3
    base_lr = 2e-4
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return lr

def get_mae_mse_loss(alpha_val):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return (1.0 - alpha_val) * mae + alpha_val * mse
    return loss

def mae_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0); y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0); y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0); y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0); y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) 
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# --- MAIN EXECUTION ---

print("Lade Daten für Cross-Validation...")
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5",
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}

X1, y1 = load_split(FILES["training"])
X2, y2 = load_split(FILES["validation"])

# Alle Slices kombinieren für sauberen CV Split
X_combined_raw = np.concatenate([X1, X2], axis=0)
y_combined_raw = np.concatenate([y1, y2], axis=0)

# Volumes erstellen
X_all, y_all = make_sliding_windows(X_combined_raw, y_combined_raw, SERIES_LEN, DEPTH)
X_all = X_all.astype(np.float32); y_all = y_all.astype(np.float32)

# CV Setup
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")



for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    RUN_NAME = f"unet_25d_MAE_MSE_CV_fold{fold+1}_alpha{ALPHA_FIXED}_{TIMESTAMP}"
    TB_ROOT = Path.home() / "data" / "tblogs_unet_3d_simple"
    TB_RUN_DIR = TB_ROOT / RUN_NAME
    TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Starte FOLD {fold+1}/{N_FOLDS} | {RUN_NAME}")

    X_train_f, y_train_f = X_all[train_idx], y_all[train_idx]
    X_val_f, y_val_f = X_all[val_idx], y_all[val_idx]

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train_f, y_train_f))
                .shuffle(len(X_train_f), reshuffle_each_iteration=True)
                .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val_f, y_val_f))
              .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
              .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model = unet_2d_stacked()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
    
    current_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1), # Changed patience from 10 --> 5
        make_epoch_ckpt_callback(RUN_NAME),
        tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"log_{RUN_NAME}.csv")),
        *tb_callbacks(TB_RUN_DIR)
    ]

    model.compile(optimizer=optimizer, loss=get_mae_mse_loss(ALPHA_FIXED), 
                  metrics=[mae_center, mse_center, psnr_center, ssim_center])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=current_callbacks)

    meta = make_meta_dict(script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                          optimizer=optimizer, learning_rate=2e-4, input_shape=(192, 240, DEPTH), # Changed learning_rate from 5e-4 --> 2e-4
                          extra={"alpha": ALPHA_FIXED, "fold": fold+1})
    
    finalize_run(model, history, RUN_NAME, meta)
    
    tf.keras.backend.clear_session()
    import gc; gc.collect()

print("\nAlle CV-Folds abgeschlossen.")