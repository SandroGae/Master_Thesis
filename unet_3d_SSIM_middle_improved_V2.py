# unet_3d_SSIM_middle_improved_V2.py
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

# REPRODUCIBILITY & DETERMINISM SETUP 
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


# Parameters
DEPTH = 3
SERIES_LEN = 41
BASEFILTERS = 64

# Simples unet in 3d
POOL_HW = (1, 2, 2)  # (D, H, W) --> Kein Pooling über depth

def conv_block_3d(x, filters, kernel_size=(3, 3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv3D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x


def unet_3d_center_output(input_shape=(DEPTH,192,240,1), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block_3d(inputs, base_filters)              ; p1 = layers.MaxPooling3D(POOL_HW)(c1)
    c2 = conv_block_3d(p1, base_filters * 2)              ; p2 = layers.MaxPooling3D(POOL_HW)(c2)
    c3 = conv_block_3d(p2, base_filters * 4)              ; p3 = layers.MaxPooling3D(POOL_HW)(c3)

    # Bottleneck
    bn = conv_block_3d(p3, base_filters * 16)

    # Decoder
    u3 = layers.Conv3DTranspose(base_filters * 4, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(bn)
    u3 = layers.Concatenate()([u3, c3])                   ; c4 = conv_block_3d(u3, base_filters * 4)
    u2 = layers.Conv3DTranspose(base_filters * 2, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])                   ; c5 = conv_block_3d(u2, base_filters * 2)
    u1 = layers.Conv3DTranspose(base_filters, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1])                   ; c6 = conv_block_3d(u1, base_filters)

    # Output Sigmoid
    out_5 = layers.Conv3D(1, (1,1,1), activation=output_activation, kernel_initializer="he_normal", use_bias=True, name="output_full")(c8)
    def take_center_slice(t):
        depth = tf.shape(t)[1]
        idx = depth // 2
        return t[:, idx:idx+1, ...]  # (B,1,H,W,1)
    out_center = layers.Lambda(take_center_slice, name="output_center")(out_5)
    return models.Model(inputs, out_center, name="unet_3d_center_only")




def load_split(h5_path):
    """
    Lädt Daten aus HDF5-Datei und formatiert sie passend für 2d unet
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]      # (H, W, N)
        high_count = f["high_count/data"][:]    # (H, W, N)

    # Achse verschieben: (H, W, N) -> (N, H, W)
    low_count = np.moveaxis(low_count, -1, 0)
    high_count = np.moveaxis(high_count, -1, 0)

    # Channel hinzufügen: (N, H, W) -> (N, H, W, C=1)
    low_count = low_count[:, :, :, np.newaxis]
    high_count = high_count[:, :, :, np.newaxis]

    return low_count, high_count


def make_sliding_windows(X, y, series_len=None, depth=None):
    """
    X, y: (N, H, W, C=1)
    return: (N_vols, Depth, H, W, C=1)
    """
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

    X_volumes = np.stack(X_volumes, axis=0)
    y_volumes = np.stack(y_volumes, axis=0)
    return X_volumes, y_volumes


def shuffle_initial(X, y, seed):
    """
    Shuffelt X und y mit der gleichen Permutation
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = np.arange(N)
    rng.shuffle(indices)
    return X[indices], y[indices]


def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    """
    Erwartet x,y je Sample als (D, H, W, C) float32.
    Schritte je Sample:
      1) Horizontal-Flip (W-Achse) mit probability p, für alle Slices identisch
      2) Clip >= 0
      3) Norm pro Slice: Division durch Summe über (H,W,C) -> Form (D,1,1,1)
      4) Zufalls-Skalierung je Sample: gleiche Skala alle Slices (Form (D,1,1,1))
    """
    def map_volume(x, y):
        # Flip, hier ist W Achse 2 (0:D, 1:H, 2:W, 3:C) und die flippen wir
        flip = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)

        # Clip >= 0
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # pro Slice normieren: (D,1,1,1)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # eine gemeinsame Skalierung für alle Slices im Volumen!
        scale = tf.random.uniform([], minval=scale_min, maxval=scale_max, dtype=tf.float32)
        x = x * scale
        y = y * scale

        return x, y
    return map_volume



# Loss
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    # erwartet (B,1,H,W,1)
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    yt = tf.squeeze(y_true, axis=1)  # (B,H,W,1)
    yp = tf.squeeze(y_pred, axis=1)
    mae = tf.reduce_mean(tf.abs(yt - yp))
    ssim_mean = tf.reduce_mean(tf.image.ssim(yt, yp, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)


# ==============================================================================
# Metriken (Angepasst: CLIPPED auf 1.0 für Vergleichbarkeit mit anderen Versionen)
# ==============================================================================
def mae_center(y_true, y_pred):
    # Alles > 1.0 wird ignoriert
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    yt = tf.squeeze(y_true, axis=1)
    yp = tf.squeeze(y_pred, axis=1)
    return tf.reduce_mean(tf.abs(yt - yp))

def mse_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    yt = tf.squeeze(y_true, axis=1)
    yp = tf.squeeze(y_pred, axis=1)
    return tf.reduce_mean(tf.math.squared_difference(yt, yp))

def psnr_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    yt = tf.squeeze(y_true, axis=1)
    yp = tf.squeeze(y_pred, axis=1)
    mse = tf.reduce_mean(tf.math.squared_difference(yt, yp), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    
    yt = tf.squeeze(y_true, axis=1)
    yp = tf.squeeze(y_pred, axis=1)
    return tf.reduce_mean(tf.image.ssim(yt, yp, max_val=1.0))
# ==============================================================================




# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
            "test":       "/home/sgaell/data/original_data/test_data.hdf5",}

BASE_NAME = "unet_3d_SSIM_middle_improved_V2"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__bf{BASEFILTERS}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Lade die Daten
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])
# X_test, y_test = load_split(FILES["test"])

# Mache daraus Volumen im Format (N_vols = 2960, DEPTH, H=192, W=240, C=1)
X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val,   y_val,   SERIES_LEN, DEPTH)

# %%
# Einmaliges initiales Shuffle (separat für Training und Validation):
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

# Batches hinzufügen
BATCH_SIZE = 8

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

# Compilieren
model = unet_3d_center_output(input_shape=(DEPTH,192,240,1))

model.compile(
    optimizer=optimizer,
    loss=mae_ssim_2d,
    metrics=[mae_center, mse_center, psnr_center, ssim_center]
)

print("Erstelle Trainingsdaten...")

AUTOTUNE = tf.data.AUTOTUNE

# Sicherstellen alles ist float 32
X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)
# X_test = X_test.astype(np.float32); y_test = y_test.astype(np.float32)

def center_target_only(x, y):
    depth = tf.shape(y)[0]
    idx = depth // 2
    return x, y[idx:idx+1, ...]  # (1,H,W,1)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(center_target_only, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(center_target_only, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))


print("Training beginnt...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    verbose=2
)

# Meta bauen
meta = make_meta_dict(
    script_name=RUN_NAME,
    batch_size=8,
    epochs=50,
    optimizer=optimizer,
    learning_rate=5e-4,
    input_shape=(DEPTH, 192, 240, 1),  # 3D-Input
    scale_range_train=(5000,15000),
    scale_range_val=(10000,10001),
    extra={"loss": "mae_ssim(alpha=0.6)", "metrics": ["mae_center","mse_center","psnr_center","ssim_center"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)

print("Training beendet...")