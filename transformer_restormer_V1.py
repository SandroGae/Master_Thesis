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
EMBED_DIM = 48  # Restormer startet meist kleiner, da er Kanäle im Encoder verdoppelt
BATCH_SIZE = 16 
INITIAL_LR = 2e-4 # Restormer verträgt etwas mehr als SRDTrans
EPOCHS = 100

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
CKPT_FOLDER = "checkpoints_transformer"

# -----------------------------
# Restormer Komponenten (Funktional)
# -----------------------------

def MDTA(x, filters, num_heads):
    """ Multi-Dconv Head Transposed Attention """
    b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 1x1 Conv für Pixel-Mix + 3x3 Depthwise Conv für lokalen Kontext
    qkv = layers.Conv2D(filters * 3, kernel_size=1, use_bias=False)(x)
    qkv = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(qkv)
    
    q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
    
    # Reshape für Transposed Attention (Attention über Kanäle C)
    # Shape: (Batch, Heads, C/Heads, H*W)
    q = tf.reshape(q, (b, h * w, num_heads, filters // num_heads))
    k = tf.reshape(k, (b, h * w, num_heads, filters // num_heads))
    v = tf.reshape(v, (b, h * w, num_heads, filters // num_heads))
    
    q = tf.transpose(q, (0, 2, 3, 1))
    k = tf.transpose(k, (0, 2, 1, 3))
    v = tf.transpose(v, (0, 2, 3, 1))
    
    # Transposed Attention Map: (C x C)
    # Das ist der Clou: Die Komplexität ist linear zur Bildgröße!
    q = tf.math.l2_normalize(q, axis=-1)
    k = tf.math.l2_normalize(k, axis=-2)
    
    attn = tf.matmul(q, k)
    attn = tf.nn.softmax(attn, axis=-1)
    
    out = tf.matmul(attn, v)
    out = tf.transpose(out, (0, 3, 1, 2))
    out = tf.reshape(out, (b, h, w, filters))
    
    out = layers.Conv2D(filters, kernel_size=1, use_bias=False)(out)
    return layers.Add()([res, out])

def GDFN(x, filters, expansion_factor=2.66):
    """ Gated-Dconv Feed-Forward Network """
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Kanäle aufblähen
    inner_filters = int(filters * expansion_factor)
    
    # Zwei Pfade für das Gating
    x = layers.Conv2D(inner_filters * 2, kernel_size=1, use_bias=False)(x)
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    
    path1, path2 = tf.split(x, num_or_size_splits=2, axis=-1)
    
    # Gating: Elementweise Multiplikation (Einer mit Aktivierung)
    x = layers.Activation('gelu')(path1) * path2
    
    # Zurück auf Ursprungskanäle
    x = layers.Conv2D(filters, kernel_size=1, use_bias=False)(x)
    return layers.Add()([res, x])

def restormer_block(x, filters, num_heads):
    x = MDTA(x, filters, num_heads)
    x = GDFN(x, filters)
    return x

# -----------------------------
# Restormer Modellaufbau (U-Net Shape)
# -----------------------------

def build_restormer(input_shape=(192, 240, 5), embed_dim=48):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Initial Embedding
    x = layers.Conv2D(embed_dim, kernel_size=3, padding='same')(inputs)
    
    # --- ENCODER ---
    # Level 1
    x1 = restormer_block(x, embed_dim, num_heads=1)
    # Downsample
    x2_in = layers.Conv2D(embed_dim * 2, kernel_size=3, strides=2, padding='same')(x1)
    
    # Level 2
    x2 = restormer_block(x2_in, embed_dim * 2, num_heads=2)
    # Downsample
    x3_in = layers.Conv2D(embed_dim * 4, kernel_size=3, strides=2, padding='same')(x2)
    
    # Level 3 (Bottleneck)
    x3 = restormer_block(x3_in, embed_dim * 4, num_heads=4)
    
    # --- DECODER ---
    # Upsample + Skip 1
    u2 = layers.Conv2DTranspose(embed_dim * 2, kernel_size=2, strides=2, padding='same')(x3)
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(embed_dim * 2, kernel_size=1)(u2) # Channel fix nach Concat
    u2 = restormer_block(u2, embed_dim * 2, num_heads=2)
    
    # Upsample + Skip 2
    u1 = layers.Conv2DTranspose(embed_dim, kernel_size=2, strides=2, padding='same')(u2)
    u1 = layers.Concatenate()([u1, x1])
    u1 = layers.Conv2D(embed_dim, kernel_size=1)(u1)
    u1 = restormer_block(u1, embed_dim, num_heads=1)
    
    # Output Refinement
    out = layers.Conv2D(embed_dim, kernel_size=3, padding='same')(u1)
    out = layers.Activation('relu')(out)
    final = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(out)
    
    return models.Model(inputs, final, name="Restormer_XRD")

# -----------------------------
# Warmup & Data Functions (Identisch zu V3)
# -----------------------------
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    return lr

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

def prepare_restormer_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) 
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# Metriken & Loss (identisch zu UNet/V3)
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_val)

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

# -----------------------------
# MAIN RUN
# -----------------------------
print("Lade Daten...")
X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
X_val, y_val = X_val.astype(np.float32), y_val.astype(np.float32)

BASE_NAME = "Restormer_XRD"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__BS{BATCH_SIZE}__seed{SEED}__emb{EMBED_DIM}__lossMAE_SSIM__{RUN_ID}"

TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)
csv_path = Path.home() / "data" / CKPT_FOLDER / f"{RUN_NAME}.csv"

# Datasets
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_restormer_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_restormer_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# Callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

model = build_restormer(embed_dim=EMBED_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

print(f"Training beginnt: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
    optimizer=optimizer, learning_rate=INITIAL_LR, input_shape=(192, 240, DEPTH),
    extra={"loss": "mae_ssim(alpha=0.6)", "model": "Restormer_XRD_Nature_Style"}
)

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print("Training beendet.")