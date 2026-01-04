#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Hilfsmodule
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# --- REPRODUZIERBARKEIT ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# --- PARAMETER ---
DEPTH = 5             # Slices (seq_length)
SERIES_LEN = 41
EMBED_DIM = 128       # Kanäle im räumlichen Teil
WINDOW_SIZE = 8       # Größe der Swin-Windows (192 und 240 sind durch 8 teilbar)
BATCH_SIZE = 8
EPOCHS = 100

# --- SWIN-HELPERS ---

def window_partition(x, window_size):
    """Teilt das Bild in lokale Fenster auf."""
    # Input: (B, H, W, C)
    _, h, w, c = x.shape
    x = tf.reshape(x, (-1, h // window_size, window_size, w // window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, (-1, window_size, window_size, c))
    return windows

def window_reverse(windows, window_size, h, w):
    """Setzt Fenster wieder zum Bild zusammen."""
    # Input: (B*num_windows, window_size, window_size, C)
    c = windows.shape[-1]
    x = tf.reshape(windows, (-1, h // window_size, w // window_size, window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (-1, h, w, c))
    return x

# --- CUSTOM LAYERS ---

class LearnedPositionalEncoding(layers.Layer):
    def __init__(self, seq_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_embeddings = self.add_weight(
            name="pos_embedding",
            shape=(1, seq_length, embedding_dim),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embeddings

class SwinTransformerBlock(layers.Layer):
    """Repliziert einen Swin-Block mit optionalem Cyclic Shift."""
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_dim = int(dim * mlp_ratio)
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(self.mlp_dim, activation=tf.nn.gelu),
            layers.Dense(dim)
        ])

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        res = x
        x = self.norm1(x)

        # Cyclic Shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))

        # Partition Windows
        x_windows = window_partition(x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, self.dim))

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, x_windows)

        # Reverse Windows
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, self.dim))
        x = window_reverse(attn_windows, self.window_size, h, w)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        x = layers.Add()([res, x])
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return layers.Add()([res, x])

# --- MODEL BUILDING ---

def build_srdtrans_swin(input_shape=(192, 240, 5), embed_dim=128):
    inputs = layers.Input(shape=input_shape)
    h, w, d = input_shape

    # 1. TEMPORAL TRANSFORMER (Pixel-wise)
    xt = layers.Reshape((h * w, d, 1))(inputs)
    xt = LearnedPositionalEncoding(seq_length=d, embedding_dim=1)(xt)
    
    # 2 Temporal Layers (Pre-Norm)
    for _ in range(2):
        res_t = xt
        xt = layers.LayerNormalization()(xt)
        xt = layers.MultiHeadAttention(num_heads=1, key_dim=4)(xt, xt)
        xt = layers.Add()([res_t, xt])
        
        res_t = xt
        xt = layers.LayerNormalization()(xt)
        xt = layers.Dense(4, activation="gelu")(xt)
        xt = layers.Dense(1)(xt)
        xt = layers.Add()([res_t, xt])

    xt = layers.Reshape((h, w, d))(xt)

    # 2. SPATIAL TRANSFORMER (Swin-Approach)
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(xt)
    
    # Paar aus W-MSA und SW-MSA
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=0)(x)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=WINDOW_SIZE // 2)(x)

    # 3. DECODER
    x = layers.Conv2D(embed_dim // 2, kernel_size=3, padding="same")(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="SRDTrans_Swin_Full")

# --- LOSS & METRIKEN ---

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_v = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_v)

def mae_center(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.clip_by_value(y_true, 0, 1) - tf.clip_by_value(y_pred, 0, 1)))

def psnr_center(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(tf.clip_by_value(y_true, 0, 1), tf.clip_by_value(y_pred, 0, 1)), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

# --- DATA PIPELINE (Dein Original-Code integriert) ---

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:] 
        high_count = f["high_count/data"][:]
    low_count = np.moveaxis(low_count, -1, 0)[:, :, :, np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[:, :, :, np.newaxis]
    return low_count, high_count

def make_sliding_windows(X, y, series_len, depth):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bX, bY = X[start:start+series_len], y[start:start+series_len]
        for s_idx in range(n_vols_per_series):
            X_v.append(bX[s_idx : s_idx + depth])
            y_v.append(bY[s_idx : s_idx + depth])
    return np.stack(X_v, axis=0), np.stack(y_v, axis=0)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x, y = x / sum_x, y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# --- MAIN ---

print("Initialisierung...")
FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5"
}

X_tr, y_tr = load_split(FILES["training"])
X_va, y_va = load_split(FILES["validation"])

X_tr, y_tr = make_sliding_windows(X_tr, y_tr, SERIES_LEN, DEPTH)
X_va, y_va = make_sliding_windows(X_va, y_va, SERIES_LEN, DEPTH)

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"SRDTrans_Swin_Final__bf{EMBED_DIM}__win{WINDOW_SIZE}__{RUN_ID}"
TB_RUN_DIR = Path.home() / "data" / "tblogs_unet_3d_simple" / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

train_ds = (tf.data.Dataset.from_tensor_slices((X_tr.astype(np.float32), y_tr.astype(np.float32)))
            .shuffle(len(X_tr), seed=SEED)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_va.astype(np.float32), y_va.astype(np.float32)))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

model = build_srdtrans_swin(input_shape=(192, 240, DEPTH), embed_dim=EMBED_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, amsgrad=True), 
              loss=mae_ssim_2d, metrics=[mae_center, psnr_center])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    make_epoch_ckpt_callback(RUN_NAME),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv")),
    *tb_callbacks(TB_RUN_DIR),
]

print(f"Training Start: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, model.optimizer, 1e-4, (192, 240, DEPTH))
finalize_run(model, history, RUN_NAME, meta)