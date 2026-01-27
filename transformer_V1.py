#!/usr/bin/env python3
# Mixed Precision aktivieren
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Compute dtype: {policy.compute_dtype}") # sollte float16 sein
print(f"Variable dtype: {policy.variable_dtype}") # sollte float32 sein

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# REPRODUZIERBARKEIT
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# PARAMETER
DEPTH = 5
SERIES_LEN = 41
EMBED_DIM = 64 
BATCH_SIZE = 8
# Dein Ziel-LR nach dem Warmup
INITIAL_LR = 5e-4 
EPOCHS = 100

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
CKPT_FOLDER = "checkpoints_transformer"

def lr_warmup_scheduler(epoch, lr):
    """ Steigert die LR über 10 Epochen linear auf INITIAL_LR """
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return INITIAL_LR * (epoch + 1) / warmup_epochs
    return lr

# SRDTRANS ARCHITEKTUR-BLÖCKE
def window_partition(x, window_size):
    s = tf.shape(x)
    b, h, w, c = s[0], s[1], s[2], s[3]
    x = tf.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, window_size, window_size, c))

def window_reverse(windows, window_size, h, w):
    total_windows = tf.shape(windows)[0]
    b = total_windows // ((h // window_size) * (w // window_size))
    x = tf.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (b, h, w, -1))

def spatio_transformer_block(x, h, w, dim, num_heads, window_size, shift_size):
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.reshape(x, (-1, h, w, dim))
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    h_pad, w_pad = h + pad_h, w + pad_w
    if shift_size > 0:
        x = tf.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
    x_windows = window_partition(x, window_size)
    x_windows = tf.reshape(x_windows, (-1, window_size * window_size, dim))
    head_dim = dim // num_heads
    qkv = layers.Dense(dim * 3)(x_windows)
    qkv = tf.reshape(qkv, (-1, window_size * window_size, 3, num_heads, head_dim))
    qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (tf.matmul(q, k, transpose_b=True)) * (head_dim ** -0.5)
    attn = tf.nn.softmax(attn, axis=-1)
    x_attn = tf.reshape(tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3)), (-1, window_size * window_size, dim))
    x_attn = layers.Dense(dim)(x_attn)
    x = tf.reshape(x_attn, (-1, window_size, window_size, dim))
    x = window_reverse(x, window_size, h_pad, w_pad)
    if shift_size > 0:
        x = tf.roll(x, shift=(shift_size, shift_size), axis=(1, 2))
    x = x[:, :h, :w, :]
    x = tf.reshape(x, (-1, h * w, dim))
    x = layers.Add()([res, x])
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(dim * 4, activation='gelu')(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([res, x])

def temporal_transformer_layer(x, seq_len, dim, num_heads):
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    pos_indices = tf.range(seq_len)[tf.newaxis, :]
    pos_embed = layers.Embedding(seq_len, dim)(pos_indices)
    x = x + pos_embed
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)(x, x)
    x = layers.Add()([res, x])
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(dim * 4, activation='gelu')(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([res, x])

def build_srdtrans(input_shape=(5, 192, 240, 1), f_maps=[16, 32, 64], window_size=7):
    inputs = layers.Input(shape=input_shape) # (B, 5, 192, 240, 1)
    x = inputs
    T0, H, W, Cin = input_shape # T0=5, H=192, W=240, Cin=1
    skip_list = [] # speichert Skip-Features aus dem Encoder

    # Encoder: 3D-Convs, aber Downsampling nur in der Zeitachse
    for filters in f_maps:
        # Feature-Extraktion (Zeit+Raum lokal, kernel=3x3x3)
        x = layers.Conv3D(filters // 2, kernel_size=(3,3,3), padding="same")(x) # (B, T, H, W, filters/2)
        x = layers.LeakyReLU(0.1)(x)

        x = layers.Conv3D(filters, kernel_size=(3,3,3), padding="same")(x) # (B, T, H, W, filters)
        x = layers.LeakyReLU(0.1)(x)

        skip_list.insert(0, x) # Skip speichern (wird im Decoder wieder addiert)

        # Zeitachse halbieren: strides=(2,1,1) => T wird kleiner, H,W bleiben gleich
        x = layers.Conv3D(filters, kernel_size=(3,3,3),strides=(2,1,1), padding="same")(x) # (B, ceil(T/2), H, W, filters)

    # Nach dem Encoder
    # Beispiel mit T0=5: T: 5 -> 3 -> 2 -> 1
    curr_t = x.shape[1]                                         # z.B. 1
    curr_c = x.shape[-1]                                        # z.B. 64

    # ------------------------------------------------------------
    # Transformer Core Teil 1: Temporal Attention pro Pixel
    # Idee: fuer jedes Pixel (h,w) wird eine Sequenz ueber die Zeit gebildet
    # ------------------------------------------------------------

    # (B, T, H, W, C) -> (B, H, W, T, C)
    x = tf.transpose(x, (0, 2, 3, 1, 4))                        # (B, H, W, T, C)

    # (B, H, W, T, C) -> (B*H*W, T, C)
    # Jedes Pixel wird zu einer "Sequenz" ueber die Zeit
    x = tf.reshape(x, (-1, curr_t, curr_c))                     # (B*H*W, T, C)

    # Transformer ueber Zeit (Self-Attention entlang T)
    x = temporal_transformer_layer(x, curr_t, curr_c, num_heads=8)  # (B*H*W, T, C)

    # Zurueck in Bildform:
    # (B*H*W, T, C) -> (B, H, W, T, C)
    x = tf.reshape(x, (-1, H, W, curr_t, curr_c))               # (B, H, W, T, C)

    # (B, H, W, T, C) -> (B, T, H, W, C)
    x = tf.transpose(x, (0, 3, 1, 2, 4))                        # (B, T, H, W, C)

    # ------------------------------------------------------------
    # Transformer Core Teil 2: Spatial Swin Blocks (Window Attention)
    # Hier wird 2D-Window-Attention auf den Pixel-Tokens gemacht
    # ------------------------------------------------------------

    # (B, T, H, W, C) -> (B*T, H*W, C)
    # weil wir spaeter in spatio_transformer_block ein (Batch, Tokens, Dim) erwarten
    x = tf.reshape(x, (-1, H * W, curr_c))                      # (B*T, H*W, C)

    # 2 Swin-Style Blocks: einmal ohne Shift, einmal mit Shift
    x = spatio_transformer_block(x, H, W, curr_c, 8, window_size, shift_size=0)                 # (B*T, H*W, C)
    x = spatio_transformer_block(x, H, W, curr_c, 8, window_size, shift_size=window_size // 2) # (B*T, H*W, C)

    # Zurueck zu (B, T, H, W, C)
    x = tf.reshape(x, (-1, curr_t, H, W, curr_c))               # (B, T, H, W, C)

    # ------------------------------------------------------------
    # Decoder: Zeitachse wieder aufblasen (Transpose-Conv in T)
    # und Skips addieren
    # ------------------------------------------------------------
    for i, filters in enumerate(reversed(f_maps)):

        # Zeit upsample: strides=(2,1,1) => T wird groesser
        x = layers.Conv3DTranspose(filters, kernel_size=(4,3,3),
                                   strides=(2,1,1), padding="same")(x)     # (B, ~2T, H, W, filters)

        # Ziel-Laenge in Zeit aus dem passenden Skip:
        target_t = skip_list[i].shape[1]                                   # z.B. 2, 3, 5

        # Falls TransposeConv zu lang ist: auf target_t croppen
        if x.shape[1] != target_t:
            x = layers.Lambda(lambda t, tt=target_t: t[:, :tt, :, :, :])(x)

        # Skip-Connection als Addition
        x = layers.Add()([x, skip_list[i]])                                # (B, target_t, H, W, filters)

        # Lokale Refinement Convs
        x = layers.Conv3D(filters // 2, 3, padding="same")(x)              # (B, target_t, H, W, filters/2)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)                   # (B, target_t, H, W, filters)
        x = layers.LeakyReLU(0.1)(x)

    # ------------------------------------------------------------
    # Output: 1 Kanal pro Frame, danach nur den mittleren Frame nehmen
    # ------------------------------------------------------------
    x = layers.Conv3D(1, 3, padding="same")(x) # (B, T0, H, W, 1)

    # Nur mittlerer Frame (bei T0=5: Index 2)
    x = x[:, T0 // 2, :, :, :]                                 # (B, H, W, 1)
    outputs = layers.Activation("sigmoid", dtype="float32", name="predictions")(x)
    return models.Model(inputs, outputs, name="SRDTrans_Paper_Exact")





# DATA UTILS
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low, high = f["low_count/data"][:], f["high_count/data"][:]
    low, high = np.moveaxis(low, -1, 0), np.moveaxis(high, -1, 0)
    return low[..., np.newaxis], high[..., np.newaxis]

def make_sliding_windows(X, y, series_len, depth):
    N = X.shape[0]
    n_series = N // series_len
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bx, by = X[start:start+series_len], y[start:start+series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s+depth]); y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, [1,2,3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, [1,2,3], keepdims=True) + 1e-12
        x, y = x/sum_x, y/sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x*scale, y*scale
    return map_vol

def prepare_input_3d(x, y):
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# LOSS & METRIKEN
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim)

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
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

# MAIN RUN
print("Lade Daten...")
X_train_raw, y_train_raw = load_split(FILES["training"])
X_val_raw, y_val_raw = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train_raw, y_train_raw, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val_raw, y_val_raw, SERIES_LEN, DEPTH)

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"SRDTrans_Exact__seed{SEED}__emb64__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train.astype('float32'), y_train.astype('float32')))
            .shuffle(len(X_train), seed=SEED)
            .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), num_parallel_calls=-1)
            .map(prepare_input_3d, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val.astype('float32'), y_val.astype('float32')))
          .map(augment_and_normalize_3d_per_slice(10000, 10001, 0), num_parallel_calls=-1)
          .map(prepare_input_3d, num_parallel_calls=-1).cache().batch(BATCH_SIZE).prefetch(-1))

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(Path.home() / "data" / CKPT_FOLDER / f"{RUN_NAME}.csv")),
    *tb_callbacks(TB_RUN_DIR)
]

model = build_srdtrans(input_shape=(DEPTH, 192, 240, 1))
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

print(f"Training beginnt: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, INITIAL_LR, (DEPTH, 192, 240, 1), 
                      extra={"model": "SRDTrans_Exact_Warmup", "depth": DEPTH})

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print("Training beendet.")