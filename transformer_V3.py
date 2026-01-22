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
BATCH_SIZE = 8
EPOCHS = 100

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}

# -----------------------------
# Daten-Pipeline (Exakt wie UNet)
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
    x = tf.transpose(x, [1, 2, 0]) # (H, W, D)
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# -----------------------------
# Loss & Metriken
# -----------------------------
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
# Swin Transformer Komponenten
# -----------------------------
def window_partition_func(x, window_size):
    # Teilt das Bild in kleine Quadrate
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, window_size * window_size, C))

def window_reverse_func(windows, window_size, h, w):
    # Setzt die Quadrate wieder zum Bild zusammen
    c = windows.shape[-1]
    x = tf.reshape(windows, (-1, h // window_size, w // window_size, window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, h, w, c))

def swin_block_functional(x, dim, num_heads, window_size, shift_size=0):
    # Ein Swin-Block ohne eine einzige Klasse!
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    res = x
    
    # 1. Normierung
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 2. Optionaler Shift (für das Paper wichtig)
    if shift_size > 0:
        x = tf.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
        
    # 3. Fenster-Partitionierung
    x_windows = window_partition_func(x, window_size) # (Num_Windows*B, 64, Dim)
    
    # 4. Attention (Hier wird nur innerhalb der 64 Pixel eines Fensters gerechnet!)
    attn_windows = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)(x_windows, x_windows)
    
    # 5. Fenster zurückbauen
    x = window_reverse_func(attn_windows, window_size, h, w)
    
    # 6. Reverse Shift
    if shift_size > 0:
        x = tf.roll(x, shift=(shift_size, shift_size), axis=(1, 2))
        
    # 7. Residual + MLP
    x = layers.Add()([res, x])
    res2 = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(dim * 4, activation="gelu")(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([res2, x])

# -----------------------------
# Das Hauptmodell (Paper-Replikation ohne Klassen)
# -----------------------------

def build_srdtrans_swin(input_shape=(192, 240, DEPTH), embed_dim=96):
    inputs = layers.Input(shape=input_shape, name="input") # (B, 192, 240, 5)
    h, w, d = input_shape

    # 1) TEMPORAL ENCODER
    # Wir wählen embed_dim so, dass sie für die Attention später passt
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(inputs)
    x = layers.ReLU()(x) # (B, 192, 240, 96)

    # 2) TEMPORAL ATTENTION (Der Fix für den Reshape-Fehler)
    # Wir projizieren die 96 Kanäle zuerst auf einen Wert, der durch d (5) teilbar ist,
    # z.B. 100 Kanäle (5 Slices * 20 Features pro Slice)
    attn_dim = 100 
    x_for_attn = layers.Dense(attn_dim)(x) 
    
    # Jetzt klappt der Reshape: (192*240, 5, 20)
    # 46080 * 5 * 20 = 4.608.000 (identisch zu 46080 * 100)
    xt = layers.Reshape((h * w, d, attn_dim // d))(x_for_attn)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, d, attn_dim // d)))(xt)
    
    res_t = xt
    xt = layers.LayerNormalization(epsilon=1e-6)(xt)
    # Attention über die 5 Slices
    xt = layers.MultiHeadAttention(num_heads=4, key_dim=(attn_dim // d) // 4)(xt, xt)
    xt = layers.Add()([res_t, xt])
    
    # Zurück in Bildform und wieder auf embed_dim (96) bringen
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, h, w, attn_dim)))(xt)
    x_spat = layers.Dense(embed_dim)(xt)

    # 3) SPATIAL SWIN BLOCKS (Räumliche Details)
    x_spat = swin_block_functional(x_spat, embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=0)
    x_spat = swin_block_functional(x_spat, embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=WINDOW_SIZE // 2)

    # 4) DECODER
    x = layers.Conv2D(embed_dim // 2, kernel_size=3, padding="same")(x_spat)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid", name="final_output")(x)

    return models.Model(inputs, outputs, name="srdtrans_v3_fixed_math")

# -----------------------------
# Main Training Run
# -----------------------------
print("Lade Daten...")
X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val, y_val = shuffle_initial(X_val, y_val, SEED)

X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
X_val, y_val = X_val.astype(np.float32), y_val.astype(np.float32)

# Run-Namen Logik aus V1 übernommen
BASE_NAME = "transformer_V3"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__emb{EMBED_DIM}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
CKPT_FOLDER = "checkpoints_transformer" # Aus V1 übernommen
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_transformer_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_transformer_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# Callbacks Logik aus V1 übernommen
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(Path.home() / "data" / CKPT_FOLDER / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

# Modell Compilieren
model = build_srdtrans_swin(embed_dim=EMBED_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d, metrics=[mae_center, mse_center, psnr_center, ssim_center])

print(f"Training beginnt: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# Meta-Daten (V1 Logik)
meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
    optimizer=optimizer, learning_rate=1e-4, input_shape=(192, 240, DEPTH),
    scale_range_train=(5000, 15000), scale_range_val=(10000, 10001),
    extra={"loss": "mae_ssim(alpha=0.6)", "model": "SRDTrans_V3_Swin"}
)

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print("Training beendet.")