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

DATA_SPLIT_SEED = 42 

DEPTH = 5
SERIES_LEN = 40 
BASEFILTERS = 64
CROP_SIZE = (512, 512)

EPOCHS = 200
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 25
RLROP_PATIENCE = 15
BATCH_SIZE = 8

# =====================================================
# ARCHITEKTUR
# =====================================================
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x


def unet_2d_stacked(input_shape=(512, 512, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
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

# =====================================================
# HILFSFUNKTIONEN DATEN (UNVERÄNDERT)
# =====================================================
def load_and_correct_danmax(base_path, scan_id, crop_size=CROP_SIZE):
    ct_file    = os.path.join(base_path, f"scan-{scan_id:04d}_orca.h5")
    white_file = os.path.join(base_path, f"scan-{scan_id-1:04d}_orca.h5")
    dark_file  = os.path.join(base_path, f"scan-{scan_id-2:04d}_orca.h5")
    data_path  = 'entry/instrument/orca/data'
    
    with h5py.File(dark_file, 'r') as f_d, h5py.File(white_file, 'r') as f_w, h5py.File(ct_file, 'r') as f_ct:
        m_dark  = np.mean(f_d[data_path][:], axis=0).astype(np.float32)
        m_white = np.mean(f_w[data_path][:], axis=0).astype(np.float32)
        
        full_h, full_w = f_ct[data_path].shape[1], f_ct[data_path].shape[2]
        h_s = (full_h - crop_size[0]) // 2
        w_s = (full_w - crop_size[1]) // 2
        
        projs = f_ct[data_path][:2000, h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]].astype(np.float32)
        m_dark_c  = m_dark[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        m_white_c = m_white[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        
        denom = m_white_c - m_dark_c
        denom[denom < 1e-6] = 1e-6
        corrected = (projs - m_dark_c) / denom
        return np.clip(corrected, 0, 1)

def load_split_danmax(base_path, gt_id, low_id):
    high_count = load_and_correct_danmax(base_path, gt_id)
    low_count  = load_and_correct_danmax(base_path, low_id)
    return low_count[..., np.newaxis], high_count[..., np.newaxis]

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

def augment_and_normalize_3d_per_slice(p: float, phys_max: float):
    def map_volume(x, y):
        # Augmentierung (Flip)
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        
        # Sicherstellen, dass keine negativen Werte durch Rauschen entstehen
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        
        # 1. Summen-Normalisierung (macht Scans vergleichbar)
        x = x / (tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12)
        y = y / (tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12)
        
        # 2. Auf den Bereich [0, 1] skalieren
        x = x / phys_max
        y = y / phys_max
        
        # 3. Finales Clipping (Sicherheit für Sigmoid)
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        
        return x, y
    return map_volume

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# =====================================================
# METRIKEN & LOSS (GEFIXT: Dynamische Max-Werte, kein Clipping)
# =====================================================
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_m = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) # Fest auf 1.0
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_m)

def ssim_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) # Fest auf 1.0

# NEU: Das ist nur für deine Augen in der Konsole (* 1000)
def display_loss_1000(y_true, y_pred):
    return mae_ssim_2d(y_true, y_pred) * 1000.0

def mae_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

def psnr_clipped(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    # Da alles zwischen 0 und 1 liegt, ist max_v einfach 1.0 (und 1.0^2 = 1.0)
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

# =====================================================
# DATEN LADEN & 60/20/20 SPLIT LOGIK
# =====================================================
print("Lade Daten von DanMAX (Bamboo)...")
BAMBOO_RAW = "../../DATA_DANMAX/2026020508/raw/bamboo/"

X_all, y_all = load_split_danmax(BAMBOO_RAW, gt_id=32, low_id=57)

print("Erstelle reproduzierbaren 60/20/20 Split auf Basis von 40er Serien...")
N_SERIES = len(X_all) // SERIES_LEN

X_series = np.reshape(X_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))
y_series = np.reshape(y_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))

rng = np.random.default_rng(DATA_SPLIT_SEED)
indices = np.arange(N_SERIES)
rng.shuffle(indices)

X_series = X_series[indices]
y_series = y_series[indices]

n_train = int(0.6 * N_SERIES)
n_val = int(0.2 * N_SERIES)

X_train_raw = np.reshape(X_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
y_train_raw = np.reshape(y_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

X_val_raw = np.reshape(X_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
y_val_raw = np.reshape(y_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

X_test_raw = np.reshape(X_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
y_test_raw = np.reshape(y_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

X_train, y_train = make_sliding_windows(X_train_raw, y_train_raw, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val_raw,   y_val_raw,   SERIES_LEN, DEPTH)
X_test,  y_test  = make_sliding_windows(X_test_raw,  y_test_raw,  SERIES_LEN, DEPTH)

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)
X_test,  y_test  = shuffle_initial(X_test,  y_test,  SEED)


# =====================================================
# DYNAMISCHE BERECHNUNG (ROBUST MIT PERCENTILEN)
# =====================================================
print("\nBerechne optimalen Skalierungsfaktor (Robust gegen Outlier)...")

def get_peak(data):
    sums = np.sum(data, axis=(2, 3, 4), keepdims=True) + 1e-12
    
    # NEU: 99.99% statt np.max()
    # Bei 512x512 Pixeln pro Bild ignorieren wir damit die extremsten ~26 Pixel pro Slice.
    # Das killt alle Hot-Pixel und extremen Artefakte, bewahrt aber das echte Signal.
    return np.percentile(data / sums, 99.99)

peak_y = get_peak(y_train)
peak_x = get_peak(X_train)

GLOBAL_PEAK_DIVISOR = max(peak_y, peak_x)

# Der Puffer bleibt, damit die 99.99% Kante nicht zu hart an der 1.0 klebt
PHYSICAL_MAX = float(GLOBAL_PEAK_DIVISOR * 1.02)

print(f"-> Robustes Peak-Niveau (99.99% Quantil): {GLOBAL_PEAK_DIVISOR:.6f}")
print(f"-> PHYSICAL_MAX wird: {PHYSICAL_MAX:.6f}\n")


# =====================================================
# TRAINING START
# =====================================================
BASE_NAME = "unet_25d_replication_V2"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME  = f"{BASE_NAME}__seed{SEED}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "scratch" / "DANMAX" / "codes" / "tb_root"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)

MODEL_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "codes" / "models"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

best_keras_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_model.keras"
best_weights_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_weights.h5"

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    
    # GEFIXT: Monitor wieder auf "val_loss" gesetzt
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_keras_file), monitor="val_loss", save_best_only=True, save_weights_only=False, mode="min", verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(best_weights_file), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE, min_lr=1e-6, verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

model = unet_2d_stacked(input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH)) 

# GEFIXT: display_loss_1000 ist jetzt als reine Ansichts-Metrik eingebaut
model.compile(
    optimizer=optimizer, 
    loss=mae_ssim_2d, 
    metrics=[display_loss_1000, mae_clipped, mse_clipped, psnr_clipped, ssim_clipped]
)

# Beispiel für train_ds (mach das analog auch für val_ds und test_ds!):
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          # HIER FEHLTE phys_max=PHYSICAL_MAX
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
          # HIER FEHLTE phys_max=PHYSICAL_MAX
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0, phys_max=PHYSICAL_MAX), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

print("Training beginnt...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

print("Evaluation auf dem Test-Set...")
test_results = model.evaluate(test_ds, verbose=1, return_dict=True)

meta = make_meta_dict(
    script_name=RUN_NAME, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    optimizer=optimizer,
    learning_rate=LR_TARGET, 
    input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH),
    extra={
        "warmup_epochs": WARMUP_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "rlrop_patience": RLROP_PATIENCE,
        "data_split_seed": DATA_SPLIT_SEED,
        "test_loss": float(test_results.get("loss", -1)),
        "test_psnr": float(test_results.get("psnr_center", -1))
    }
)

finalize_run(model, history, RUN_NAME, meta)