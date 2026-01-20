# cross_val_unet_25d_improved_V2.py
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

# Reproduzierbatkeit
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
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# --- Funktionen (Modell, Data Loading, Metriken) bleiben identisch ---

def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, base_filters) ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2) ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4) ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8) ; p4 = layers.MaxPooling2D((2, 2))(c4)

    bn = conv_block_2d(p4, base_filters * 16)

    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]) ; c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]) ; c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]) ; c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]) ; c8 = conv_block_2d(u1, base_filters)
    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8)
    return models.Model(inputs, out, name="unet_25d_stacked")

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]
        high_count = f["high_count/data"][:]
    low_count = np.moveaxis(low_count, -1, 0)[:, :, :, np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[:, :, :, np.newaxis]
    return low_count, high_count

def make_sliding_windows(X, y, series_len=None, depth=None):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_volumes, y_volumes = [], []
    for i in range(n_series):
        start = i * series_len
        blockX, blockY = X[start:start+series_len], y[start:start+series_len]
        for start_idx in range(n_vols_per_series):
            X_volumes.append(blockX[start_idx : start_idx + depth])
            y_volumes.append(blockY[start_idx : start_idx + depth])
    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x, y = x / sum_x, y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_mean = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)

def mae_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0])
    idx = tf.shape(y)[0] // 2
    return x, y[idx]



# DATEN LADEN & K-FOLD LOOP 
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5",
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}

X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

# Alle Daten kombinieren
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)

num_series = len(X_all) // SERIES_LEN
series_indices = np.arange(num_series)

BASE_NAME = "cross_val_unet_25d_improved_V2"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
TB_ROOT = Path.home() / "data" / "tblogs_unet_3d_simple"
all_fold_scores = []

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(series_indices)):
    fold_id = fold + 1
    print(f"\n--- STARTE FOLD {fold_id} ---")
    
    # Fold-spezifische Pfade
    FOLD_NAME = f"{BASE_NAME}_fold{fold_id}_{RUN_ID}"
    FOLD_DIR = TB_ROOT / FOLD_NAME
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Daten extrahieren (auf Serien-Ebene!)
    def get_fold_data(indices):
        X_list, y_list = [], []
        for i in indices:
            start = i * SERIES_LEN
            X_list.append(X_all[start : start + SERIES_LEN])
            y_list.append(y_all[start : start + SERIES_LEN])
        return np.concatenate(X_list), np.concatenate(y_list)

    X_train_f, y_train_f = get_fold_data(train_idx)
    X_val_f, y_val_f = get_fold_data(val_idx)

    # Sliding Windows & Shuffle
    X_train_f, y_train_f = make_sliding_windows(X_train_f, y_train_f, SERIES_LEN, DEPTH)
    X_val_f, y_val_f = make_sliding_windows(X_val_f, y_val_f, SERIES_LEN, DEPTH)
    X_train_f, y_train_f = shuffle_initial(X_train_f, y_train_f, SEED)
    X_val_f, y_val_f = shuffle_initial(X_val_f, y_val_f, SEED)

    # Modell erstellen
    model = unet_2d_stacked(input_shape=(192, 240, DEPTH))
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
    model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

    # Datasets
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train_f.astype(np.float32), y_train_f.astype(np.float32)))
                .shuffle(len(X_train_f), seed=SEED, reshuffle_each_iteration=True)
                .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val_f.astype(np.float32), y_val_f.astype(np.float32)))
              .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

    # Training
    fold_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        make_epoch_ckpt_callback(FOLD_NAME),
        tf.keras.callbacks.CSVLogger(str(FOLD_DIR / f"{FOLD_NAME}.csv")),
        *tb_callbacks(FOLD_DIR)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=fold_callbacks, verbose=2)

    # Finalisieren & Metriken sammeln
    all_fold_scores.append(min(history.history['val_mae_center']))
    meta = make_meta_dict(FOLD_NAME, BATCH_SIZE, 100, optimizer, 5e-4, (192,240,DEPTH), (5000,15000), (10000,10001))
    finalize_run(model, history, FOLD_NAME, meta)
    
    tf.keras.backend.clear_session()

print(f"\nK-Fold abgeschlossen. Durchschnittlicher MAE: {np.mean(all_fold_scores):.6f}")