# unet_3d_SSIM.py
#!/usr/bin/env python3

# import os
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import h5py
import numpy as np
from datetime import datetime

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks


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

def unet_3d(input_shape=(DEPTH, 192, 240, 1), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block_3d(inputs, base_filters)              ; p1 = layers.MaxPooling3D(POOL_HW)(c1)
    c2 = conv_block_3d(p1, base_filters * 2)              ; p2 = layers.MaxPooling3D(POOL_HW)(c2)
    c3 = conv_block_3d(p2, base_filters * 4)              ; p3 = layers.MaxPooling3D(POOL_HW)(c3)
    c4 = conv_block_3d(p3, base_filters * 8)              ; p4 = layers.MaxPooling3D(POOL_HW)(c4)

    # Bottleneck
    bn = conv_block_3d(p4, base_filters * 16)

    # Decoder
    u4 = layers.Conv3DTranspose(base_filters * 8, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])                   ; c5 = conv_block_3d(u4, base_filters * 8)

    u3 = layers.Conv3DTranspose(base_filters * 4, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])                   ; c6 = conv_block_3d(u3, base_filters * 4)

    u2 = layers.Conv3DTranspose(base_filters * 2, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])                   ; c7 = conv_block_3d(u2, base_filters * 2)

    u1 = layers.Conv3DTranspose(base_filters, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])                   ; c8 = conv_block_3d(u1, base_filters)

    # Output Sigmoid
    out = layers.Conv3D(1, (1, 1, 1), activation=output_activation, kernel_initializer="he_normal", use_bias=True, name="output")(c8)

    return models.Model(inputs, out, name="unet_3d_simple_relu_sigmoid")




# ==== (SSIM für 3D via Slices) ============================================
def combined_mae_ssim_3d(y_true, y_pred, alpha=0.6):
    """
    Kombi-Loss für 3D-Volumes über 2D-SSIM pro Slice:
      Loss = (1-alpha)*MAE + alpha*(1-SSIM_mean)
    Erwartet: Inputs in Form (B, D, H, W, C)
    Gibt: Skalar (Batch-Loss)
    """
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)

    # Reshape 5d --> 4d: (B,D,H,W,C) -> (B*D, H, W, C)
    shape = tf.shape(y_true)
    B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    y_true_4d = tf.reshape(y_true, (B*D, H, W, C))
    y_pred_4d = tf.reshape(y_pred, (B*D, H, W, C))

    # SSIM pro Slice, dann Mittelwert über alle Slices im Batch
    ssim_vals = tf.image.ssim(y_true_4d, y_pred_4d, max_val=1.0)
    ssim_mean = tf.reduce_mean(ssim_vals)

    # MAE über alle Voxels im Batch
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)
# ==============================================================================

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
    x,y: (D, H, W, C)
    1) optional flip
    2) relu
    3) pro Slice durch Summe normalisieren
    4) eine gemeinsame (!) Skalierung fuer das ganze Volume
    """
    def map_volume(x, y):
        # optional flip
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)

        # negativ raus
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # pro Slice normieren
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # gemeinsame Skalierung
        scale = tf.random.uniform([], minval=scale_min, maxval=scale_max, dtype=tf.float32)
        x = x * scale
        y = y * scale

        # optional leicht clampen
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)

        return x, y
    return map_volume





# Metriken
def psnr_metric_3d_per_sample(y_true, y_pred):
    # y_true/y_pred: (N, D, H, W, C) in [0,1]
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3,4))  # MSE gemittelt über (D, H, W, C)
    psnr = 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)               # PSNR
    return psnr  # Mittelwert über N

def ssim_3d_metric(y_true, y_pred):
    # gleiche Logik wie oben, nur der Mittelwert von SSIM als Metrik
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    B, D, H, W, C = tf.unstack(tf.shape(y_true))
    yt4 = tf.reshape(y_true, (B*D, H, W, C))
    yp4 = tf.reshape(y_pred, (B*D, H, W, C))
    return tf.reduce_mean(tf.image.ssim(yt4, yp4, max_val=1.0))

# ===== Center-Slice Metrics (nur mittleres Bild) =====
def _center_hw(y_true, y_pred):
    # Erwartet: (B, D, H, W, C) in [0,1]
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    D = tf.shape(y_true)[1]
    idx = D // 2  # robust, auch wenn DEPTH mal != 5 ist
    yt = y_true[:, idx, :, :, :]  # (B, H, W, C)
    yp = y_pred[:, idx, :, :, :]  # (B, H, W, C)
    return yt, yp

def mae_center_slice(y_true, y_pred):
    yt, yp = _center_hw(y_true, y_pred)
    return tf.reduce_mean(tf.abs(yt - yp))

def mse_center_slice(y_true, y_pred):
    yt, yp = _center_hw(y_true, y_pred)
    return tf.reduce_mean(tf.math.squared_difference(yt, yp))

def psnr_center_slice(y_true, y_pred):
    yt, yp = _center_hw(y_true, y_pred)
    # PSNR über Batch mitteln
    mse = tf.reduce_mean(tf.math.squared_difference(yt, yp), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center_slice(y_true, y_pred):
    yt, yp = _center_hw(y_true, y_pred)
    return tf.reduce_mean(tf.image.ssim(yt, yp, max_val=1.0))



# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
            "test":       "/home/sgaell/data/original_data/test_data.hdf5",}


BASE_NAME = "unet_3d_SSIM_test"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__bf{BASEFILTERS}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR = make_run_dir(RUN_NAME, root=TB_ROOT)

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
    *tb_callbacks(TB_RUN_DIR, histograms=False, profile=False),
]

# Compilieren
model = unet_3d(input_shape=(DEPTH, 192, 240, 1))
model.compile(
    optimizer=optimizer,
    loss=lambda yt, yp: combined_mae_ssim_3d(yt, yp, alpha=0.6), # kombinierter Loss (60% SSIM, 40% MAE)
    metrics=['mae', 'mse', psnr_metric_3d_per_sample, ssim_3d_metric, mae_center_slice, mse_center_slice, psnr_center_slice, ssim_center_slice]
)

print("Erstelle Trainingsdate...")

AUTOTUNE = tf.data.AUTOTUNE

# Sicherstellen alles ist float 32
X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)
# X_test = X_test.astype(np.float32); y_test = y_test.astype(np.float32)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(0.5, 1.5, p=0.5), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(1.0, 1.0, p=0.0), num_parallel_calls=AUTOTUNE)
          .cache() 
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))


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
    scale_range_train=(0.5, 1.5),
    scale_range_val=(1.0, 1.0),
    extra={"loss": "mae_ssim(alpha=0.6)", "metrics": ["mae", "mse", "psnr_metric_3d_per_sample"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)

print("Training beendet...")
