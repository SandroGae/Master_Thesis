# cross_val_unet_25d_SSIM_middle_improved_V2_interpolated_SEED43.py
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
from tqdm import tqdm

from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# --- REPRODUZIERBARKEIT ---
DATA_SPLIT_SEED = 42  # Bleibt 42, damit Fold 2 dieselben Patienten enthält
INIT_SEED = 42        # Neuer Seed

os.environ['PYTHONHASHSEED'] = str(DATA_SPLIT_SEED)
random.seed(DATA_SPLIT_SEED)
np.random.seed(DATA_SPLIT_SEED)
# Globaler TF Seed wird unten im Loop spezifisch für die Initialisierung gesetzt
tf.config.experimental.enable_op_determinism()

# Konfiguration
DEPTH = 5
MAX_STRIDE = 24
USE_POISSON_NOISE = True
SERIES_LEN_INTERP = 241
SERIES_LEN_ORIG   = 41
BASEFILTERS       = 64
BATCH_SIZE        = 8
AUTOTUNE          = tf.data.AUTOTUNE

# Pfade
DATA_ROOT = Path.home() / "data"
INTERP_DIR = DATA_ROOT / "interpolated_data_linear"
ORIG_DIR   = Path.home() / "data/original_data"

suffix = "pois_on.hdf5" if USE_POISSON_NOISE else "pois_off.hdf5"
TRAIN_FILE = INTERP_DIR / f"interpolated_training_data_{suffix}"

# --- FUNKTIONEN ---

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
        low_ds = f["low_count/data"]
        high_ds = f["high_count/data"]
        num_imgs = low_ds.shape[-1]
        h, w = low_ds.shape[0], low_ds.shape[1]
        low_count = np.empty((num_imgs, h, w, 1), dtype=np.float32)
        high_count = np.empty((num_imgs, h, w, 1), dtype=np.float32)
        print(f"Lade {h5_path}...")
        pbar = tqdm(total=num_imgs, unit="Bilder", desc="RAM Loading")
        chunk_size = 100
        for start in range(0, num_imgs, chunk_size):
            end = min(start + chunk_size, num_imgs)
            low_count[start:end, ..., 0] = np.moveaxis(low_ds[..., start:end], -1, 0)
            high_count[start:end, ..., 0] = np.moveaxis(high_ds[..., start:end], -1, 0)
            pbar.update(end - start)
        pbar.close()
    return low_count, high_count

def make_strided_windows(X, y, series_len, depth, stride, step=1):
    N, H, W, C = X.shape
    n_series = N // series_len
    span_needed = (depth - 1) * stride + 1
    n_vols_per_series = series_len - span_needed + 1
    X_vols, y_vols = [], []
    for i in range(n_series):
        base = i * series_len
        bX, bY = X[base:base+series_len], y[base:base+series_len]
        for start_idx in range(0, n_vols_per_series, step):
            indices = np.arange(start_idx, start_idx + span_needed, stride)
            if indices[-1] >= series_len: continue
            X_vols.append(bX[indices])
            y_vols.append(bY[indices])
    return (np.stack(X_vols, axis=0), np.stack(y_vols, axis=0)) if X_vols else (np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C)))

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0])
    idx = tf.shape(y)[0] // 2
    return x, y[idx]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, [1,2,3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, [1,2,3], keepdims=True) + 1e-12
        scale = tf.random.uniform([], scale_min, scale_max)
        return (x / sum_x) * scale, (y / sum_y) * scale
    return map_volume


def lr_warmup_func(epoch, lr):
    target_lr = 5e-4
    warmup_epochs = 3
    
    if epoch < warmup_epochs:
        # Berechnet die LR für die ersten 3 Epochen
        new_lr = (epoch + 1) * (target_lr / warmup_epochs)
        return new_lr
    return lr

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim)

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

# --- DATEN LADEN ---
X_interp_raw, y_interp_raw = load_split(TRAIN_FILE)
TRAIN_ORIG_FILE = Path.home() / "data/original_data/training_data.hdf5"
X_orig_raw, y_orig_raw = load_split(TRAIN_ORIG_FILE)

num_series = len(X_orig_raw) // SERIES_LEN_ORIG
series_indices = np.arange(num_series)

kf = KFold(n_splits=5, shuffle=True, random_state=DATA_SPLIT_SEED)

# --- FOLD LOOP ---
for fold, (train_idx, val_idx) in enumerate(kf.split(series_indices)):
    fold_id = fold + 1
    
    # GEZIELTES TRAINING NUR FÜR FOLD 2
    if fold_id != 2:
        continue
        
    print(f"\n{'='*60}\nRE-RUN FOLD {fold_id} | INIT-SEED: {INIT_SEED}\n{'='*60}")
    
    # Hier setzen wir den Initialisierungs-Seed für das Netzwerk
    tf.random.set_seed(INIT_SEED)

    BASE_NAME = "fold2_only_unet_25d_SSIM_middle_improved_V2_interpolated"
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    FOLD_NAME = f"{BASE_NAME}_fold{fold_id}_SEED{INIT_SEED}_{RUN_ID}"
    FOLD_DIR = Path.home() / "data" / "tblogs_unet_3d_simple" / FOLD_NAME
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Daten extrahieren
    X_tr_fold = np.concatenate([X_interp_raw[i*SERIES_LEN_INTERP : (i+1)*SERIES_LEN_INTERP] for i in train_idx])
    y_tr_fold = np.concatenate([y_interp_raw[i*SERIES_LEN_INTERP : (i+1)*SERIES_LEN_INTERP] for i in train_idx])
    X_va_fold = np.concatenate([X_orig_raw[i*SERIES_LEN_ORIG : (i+1)*SERIES_LEN_ORIG] for i in val_idx])
    y_va_fold = np.concatenate([y_orig_raw[i*SERIES_LEN_ORIG : (i+1)*SERIES_LEN_ORIG] for i in val_idx])

    # 2. Fensterbau
    X_tr_win_list, y_tr_win_list = [], []
    for s in [1, 2, 4, 6, 12, 24]:
        step = 5 if s == 12 else (4 if s == 24 else 6)
        Xw, yw = make_strided_windows(X_tr_fold, y_tr_fold, SERIES_LEN_INTERP, DEPTH, stride=s, step=step)
        X_tr_win_list.append(Xw.astype(np.float32))
        y_tr_win_list.append(yw.astype(np.float32))
    
    X_tr_win = np.concatenate(X_tr_win_list, axis=0)
    y_tr_win = np.concatenate(y_tr_win_list, axis=0)
    X_va_win, y_va_win = make_strided_windows(X_va_fold, y_va_fold, SERIES_LEN_ORIG, DEPTH, stride=1)

    X_tr_win, y_tr_win = shuffle_initial(X_tr_win, y_tr_win, DATA_SPLIT_SEED)
    X_va_win, y_va_win = shuffle_initial(X_va_win, y_va_win, DATA_SPLIT_SEED)

    # 3. Training
    model = unet_2d_stacked(input_shape=(192, 240, DEPTH))
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
    model.compile(optimizer=opt, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr_win, y_tr_win))
                .shuffle(1000, seed=DATA_SPLIT_SEED, reshuffle_each_iteration=True)
                .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_va_win, y_va_win))
              .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

    callbacks = [
        # Nutzt die oben definierte Funktion
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_func, verbose=1),
        
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        make_epoch_ckpt_callback(FOLD_NAME),
        tf.keras.callbacks.CSVLogger(str(FOLD_DIR / f"{FOLD_NAME}.csv")),
        *tb_callbacks(FOLD_DIR)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks, verbose=2)

    finalize_run(model, history, FOLD_NAME, make_meta_dict(FOLD_NAME, BATCH_SIZE, 100, opt, 5e-4, (192,240,DEPTH)))
    tf.keras.backend.clear_session()

print("\nFold 2 Re-Run abgeschlossen.")