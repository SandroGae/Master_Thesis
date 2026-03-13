#25D_replication.py
#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

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
SERIES_LEN = 40  # Geändert auf 40, damit 2000 Bilder glatt durch 40 teilbar sind
BASEFILTERS = 64
CROP_SIZE = (512, 512) # Verhindert Out-of-Memory (OOM)

# Simples unet in 2.5D
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(512, 512, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
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

# --- NEUE HILFSFUNKTION FÜR DANMAX ---
def load_and_correct_danmax(base_path, scan_id, crop_size=CROP_SIZE):
    ct_file    = os.path.join(base_path, f"scan-{scan_id:04d}_orca.h5")
    white_file = os.path.join(base_path, f"scan-{scan_id-1:04d}_orca.h5")
    dark_file  = os.path.join(base_path, f"scan-{scan_id-2:04d}_orca.h5")
    data_path  = 'entry/instrument/orca/data'
    
    with h5py.File(dark_file, 'r') as f_d, h5py.File(white_file, 'r') as f_w, h5py.File(ct_file, 'r') as f_ct:
        m_dark  = np.mean(f_d[data_path][:], axis=0).astype(np.float32)
        m_white = np.mean(f_w[data_path][:], axis=0).astype(np.float32)
        projs   = f_ct[data_path][:2000].astype(np.float32) # Nimm 2000 Bilder
        
        denom = m_white - m_dark
        denom[denom < 1e-6] = 1e-6
        corrected = (projs - m_dark) / denom
        corrected = np.clip(corrected, 0, 1)
        
        h_s = (corrected.shape[1] - crop_size[0]) // 2
        w_s = (corrected.shape[2] - crop_size[1]) // 2
        return corrected[:, h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]

def load_split_danmax(base_path, gt_id, low_id):
    high_count = load_and_correct_danmax(base_path, gt_id)
    low_count  = load_and_correct_danmax(base_path, low_id)
    
    # Add Channel axis: (N, H, W) -> (N, H, W, 1)
    return low_count[..., np.newaxis], high_count[..., np.newaxis]

# --- RESTLICHER CODE IDENTISCH ---
def make_sliding_windows(X, y, series_len=None, depth=None):
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_vols, y_vols = [], []
    for i in range(n_series):
        start = i * series_len
        bx, by = X[start:start+series_len], y[start:start+series_len]
        for s_idx in range(n_vols_per_series):
            X_vols.append(bx[s_idx : s_idx + depth])
            y_vols.append(by[s_idx : s_idx + depth])
    return np.stack(X_vols, axis=0), np.stack(y_vols, axis=0)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x; y = y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_m = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_m)

def mae_center(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.clip_by_value(y_true, 0, 1) - tf.clip_by_value(y_pred, 0, 1)))

def mse_center(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(tf.clip_by_value(y_true, 0, 1) - tf.clip_by_value(y_pred, 0, 1)))

def psnr_center(y_true, y_pred):
    y_t = tf.clip_by_value(y_true, 0, 1); y_p = tf.clip_by_value(y_pred, 0, 1)
    mse = tf.reduce_mean(tf.math.squared_difference(y_t, y_p), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.clip_by_value(y_true, 0, 1), tf.clip_by_value(y_pred, 0, 1), 1.0))

# --- TRAINING START ---
print("Lade Daten von DanMAX (Bamboo)...")
BAMBOO_RAW = "../../DATA_DANMAX/2026020508/raw/bamboo/"

# Wir laden den gesamten Stack (2000 Bilder). 
# Da das viel RAM braucht, machen wir den Split manuell:
X_all, y_all = load_split_danmax(BAMBOO_RAW, gt_id=32, low_id=57)

# Split 80/20 für Training/Validation
split_idx = int(0.8 * len(X_all))
X_train_raw, y_train_raw = X_all[:split_idx], y_all[:split_idx]
X_val_raw,   y_val_raw   = X_all[split_idx:], y_all[split_idx:]

X_train, y_train = make_sliding_windows(X_train_raw, y_train_raw, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val_raw,   y_val_raw,   SERIES_LEN, DEPTH)

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

BASE_NAME = "unet_25d_replication"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME  = f"{BASE_NAME}__seed{SEED}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
LR_TARGET = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)

# --- DER EINZIGE SPEICHERORT FÜR MODELLE ---
MODEL_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "codes" / "models"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

best_keras_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_model.keras"
best_weights_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_weights.h5"

# --- CALLBACKS DEFINIEREN ---
callbacks = [
    # 1. Speichert das komplette Modell (.keras)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_keras_file),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    ),
    # 2. Speichert NUR die Gewichte (.h5)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_weights_file),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0
    ),
    # Trainings-Steuerung (Plateau & Early Stopping)
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    
    # Deine Custom Logging Callbacks
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

model = unet_2d_stacked(input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH)) 
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

print("Training beginnt...")
history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, verbose=2)

meta = make_meta_dict(script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=200, optimizer=optimizer,
                      learning_rate=LR_TARGET, input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH))
finalize_run(model, history, RUN_NAME, meta)