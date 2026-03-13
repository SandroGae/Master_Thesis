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

# =====================================================
# PARAMETER CONFIGURATION
# =====================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64

# Training Hyperparameter
EPOCHS = 200
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 25
RLROP_PATIENCE = 15
BATCH_SIZE = 8

# =====================================================
# ARCHITEKTUR & HILFSFUNKTIONEN
# =====================================================
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    c1 = conv_block_2d(inputs, base_filters)          ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2)          ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4)          ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8)          ; p4 = layers.MaxPooling2D((2, 2))(c4)

    bn = conv_block_2d(p4, base_filters * 16)

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

    X_volumes, y_volumes = [], []
    for i in range(0, n_series, 1):
        start = i * series_len
        blockX = X[start:start+series_len]
        blockY = y[start:start+series_len]

        for start_idx in range(0, n_vols_per_series, 1):
            X_volumes.append(blockX[start_idx : start_idx + depth])
            y_volumes.append(blockY[start_idx : start_idx + depth])

    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

# =====================================================
# DYNAMISCHE SKALIERUNG & METRIKEN
# =====================================================
def get_dynamic_max(data_array):
    """Ermittelt automatisch die wahrscheinliche Bit-Tiefe anhand des Maximalwerts."""
    m = np.max(data_array)
    if m <= 255: return 255.0         # 8-bit
    elif m <= 4095: return 4095.0     # 12-bit
    elif m <= 65535: return 65535.0   # 16-bit
    else: return float(m)             # Fallback

# UPDATE: Nimmt jetzt getrennte Max-Werte für X und Y
def normalize_v2_minmax(max_x: float, max_y: float, p: float = 0.5):
    def map_volume(x, y):
        flip = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)

        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # Separate Skalierung für LC (X) und GT (Y)
        x = x / max_x
        y = y / max_y

        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        return x, y
    return map_volume

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)       
    x = tf.transpose(x, [1, 2, 0])   
    depth = tf.shape(y)[0]
    idx = depth // 2
    y_center = y[idx]                
    return x, y_center

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_mean = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)

def display_loss(y_true, y_pred):
    return mae_ssim_2d(y_true, y_pred) * 10000.0

def mae_center(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.clip_by_value(y_true, 0.0, 1.0) - tf.clip_by_value(y_pred, 0.0, 1.0)))
def mse_center(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)))
def psnr_center(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)
def ssim_center(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0), max_val=1.0))

def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS: return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

# =====================================================
# DATEN LADEN & PIPELINE
# =====================================================
print("Lade Daten...")
FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}

X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val,   y_val,   SERIES_LEN, DEPTH)

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)

# --- NEU: Dynamische Bestimmung der Skalierung ---
MAX_X = get_dynamic_max(X_train)
MAX_Y = get_dynamic_max(y_train)
print(f"Automatisches Scaling erkannt: LC (X) max={MAX_X}, GT (Y) max={MAX_Y}")

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(normalize_v2_minmax(max_x=MAX_X, max_y=MAX_Y, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(normalize_v2_minmax(max_x=MAX_X, max_y=MAX_Y, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# =====================================================
# TRAINING START
# =====================================================
BASE_NAME = "unet_25d_V2_pure_scaling"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME  = f"{BASE_NAME}__seed{SEED}__bf{BASEFILTERS}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "codes" / "models"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
best_keras_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_model.keras"
best_weights_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_weights.h5"

optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_keras_file), monitor="val_display_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_weights_file), monitor="val_display_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_display_loss", factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-6, verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_display_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=str(MODEL_OUT_DIR)),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

model = unet_2d_stacked(input_shape=(192, 240, DEPTH)) 
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[display_loss, mae_center, mse_center, psnr_center, ssim_center])

print("Training beginnt...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, optimizer=optimizer,
    learning_rate=LR_TARGET, input_shape=(192, 240, DEPTH),  
    extra={
        "warmup_epochs": WARMUP_EPOCHS, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "rlrop_patience": RLROP_PATIENCE, "max_val_X": MAX_X, "max_val_Y": MAX_Y,
        "normalization": f"minmax(X={MAX_X}, Y={MAX_Y})", "loss": "mae_ssim(alpha=0.6)"
    }
)
finalize_run(model, history, RUN_NAME, meta)
print("Training beendet...")