#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Importiere deine Hilfsskripte (müssen im gleichen Ordner liegen oder im Python Path)
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks


# Reproduzierbatkeit
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


# Konfiguration
DEPTH = 5 # Volumentiefe
MAX_STRIDE = 24 # Data Augmentation: Strides von 1 bis MAX_STRIDE
USE_POISSON_NOISE = True  # True = ...pois_on.hdf5, False = ...pois_off.hdf5

# Basis-Parameter
SERIES_LEN_INTERP = 241  # Länge der interpolierten Serien (41 + 40 * 5)
SERIES_LEN_ORIG   = 41   # Länge der Original-Serien (Validation)
BASEFILTERS       = 64

# Pfade (Auf Scratch anpassen!)
DATA_ROOT = Path.home() / "data"
INTERP_DIR = DATA_ROOT / "interpolated_data_linear"
ORIG_DIR    = Path.home() / "data/original_data"

# Dateinamen bauen
if USE_POISSON_NOISE == True:
    suffix = "pois_on.hdf5"
else:
    suffix = "pois_off.hdf5"
TRAIN_FILE = INTERP_DIR / f"interpolated_training_data_{suffix}" # Interpolierte Trainingsdaten
VAL_FILE   = ORIG_DIR / "validation_data.hdf5" # Originaldaten ohne Interpolation


# Simples unet in 2.5D
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape, base_filters=BASEFILTERS, output_activation="sigmoid"):
    # Input Shape ist (H, W, DEPTH) -> Depth wird als Channel behandelt
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

    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8) # Direkt 1 Channel (kein Lambda Slicing mehr nötig)
    
    return models.Model(inputs, out, name=f"unet_25d_stacked_D{DEPTH}")



def load_split(h5_path):
    """
    Lädt Daten aus HDF5-Datei und formatiert sie passend für 2.5d unet
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]      # (H, W, N)
        high_count = f["high_count/data"][:]    # (H, W, N)

    low_count = np.moveaxis(low_count, -1, 0) # Achse verschieben: (H, W, N) -> (N, H, W)
    high_count = np.moveaxis(high_count, -1, 0)

    low_count = low_count[:, :, :, np.newaxis] # Channel hinzufügen: (N, H, W) -> (N, H, W, C=1)
    high_count = high_count[:, :, :, np.newaxis]

    return low_count.astype(np.float16), high_count.astype(np.float16)


def make_strided_windows(X, y, series_len, depth, stride, step=1):
    """
    Erstellt Volumina mit spezifischem Stride und spezifischen steps
    step=1: Jedes mögliche Fenster (gleitend)
    step=3: Jedes dritte Fenster (ausgedünnt)
    """
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    
    # Berechne, wie weit das Fenster greift (Spanne)
    span_needed = (depth - 1) * stride + 1
    
    # Wie viele Fenster passen in eine Serie?
    n_vols_per_series = series_len - span_needed + 1

    if n_vols_per_series <= 0:
        return np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C))

    X_volumes = []
    y_volumes = []

    for i in range(n_series):
        base = i * series_len
        blockX = X[base : base+series_len]
        blockY = y[base : base+series_len]

        # Nutze 'step' im range(start, stop, step)
        for start_idx in range(0, n_vols_per_series, step):
            indices = np.arange(start_idx, start_idx + span_needed, stride)
            
            # Safety Check
            if indices[-1] >= series_len:
                continue
                
            X_volumes.append(blockX[indices])
            y_volumes.append(blockY[indices])

    if len(X_volumes) == 0:
        return np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C))
        
    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)


def shuffle_initial(X, y, seed):
    """Shuffelt X und y 'in-place' ohne Speicher-Kopie."""
    seed_seq = np.random.SeedSequence(seed)
    rng1 = np.random.default_rng(seed_seq)
    rng2 = np.random.default_rng(seed_seq)
    rng1.shuffle(X)
    rng2.shuffle(y)
    return X, y

def cast_to_float32(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)


def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    """
    Erwartet x,y je Sample als (D, H, W, C) float32.
    Schritte je Sample:
      1) Horizontal-Flip (W-Achse) mit probability p, für alle Slices identisch
      2) Clip >= 0
      3) Norm pro Slice: Division durch Summe über (H,W,C) -> Form (D,1,1,1)
      4) Zufalls-Skalierung je Sample: gleiche Skala alle Slices (Form (D,1,1,1))
    """
    def map_volume(x, y):
        # W Achse ist 2 (0:D, 1:H, 2:W, 3:C), die flippen
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)

        # Clip >= 0
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # pro Slice normieren: (D,1,1,1)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # eine gemeinsame Skalierung für alle Slices im Volumen!
        scale = tf.random.uniform([], scale_min, scale_max)
        x = x * scale
        y = y * scale
        return x, y
    
    return map_volume

# Loss
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    # Input ist jetzt direkt (B, H, W, 1) -> Kein Squeeze mehr nötig
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_mean = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)

# Metriken (Angepasst: CLIPPED auf 1.0 für Vergleichbarkeit)
def mae_center(y_true, y_pred):
    # Alles > 1.0 wird ignoriert
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    # MSE über (H, W, C)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))



print(f"Lade Daten: {TRAIN_FILE}")
X_train_raw, y_train_raw = load_split(TRAIN_FILE)

X_train_list = []
y_train_list = []

# Strides wählen
SELECTED_STRIDES = [1, 2, 4, 6, 12, 24]

print(f"Generiere Volumina für Strides: {SELECTED_STRIDES}...")

 # Logik für sinnvolle Step Werte um RAM zu reduzieren
for stride_size in SELECTED_STRIDES:
    if stride_size == 1:
        current_step = 6
    elif stride_size == 2:
        current_step = 4
    elif stride_size == 6:
        current_step = 1
    else: # Für 4, 12 ,24
        current_step = 2
    print(f" -> Verarbeite Stride {stride_size} mit Step {current_step}...")

    X_vol, y_vol = make_strided_windows(
        X_train_raw, 
        y_train_raw, 
        SERIES_LEN_INTERP, 
        DEPTH, 
        stride=stride_size, 
        step=current_step
    )
    
    if len(X_vol) > 0:
        X_train_list.append(X_vol)
        y_train_list.append(y_vol)
        print(f"    Ergebnis: {len(X_vol)} Volumen.")
    else:
        print(f"    Warnung: Keine Volumen für Stride {stride_size} (zu groß für Serie).")

# Memory freigeben (X_train_raw redundant)
del X_train_raw, y_train_raw

print("Concatenating Training Data...")
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

# Memory freigeben
del X_train_list, y_train_list

print(f"SHUFFLE Training Data ({len(X_train)} Volumen total)...")
X_train, y_train = shuffle_initial(X_train, y_train, SEED)


# Validierungsdaten original
print(f"Lade Validation Raw: {VAL_FILE}")
X_val_raw, y_val_raw = load_split(VAL_FILE)
print("Generiere Validation Volumen (Stride 1 auf Original)...")
X_val, y_val = make_strided_windows(X_val_raw, y_val_raw, SERIES_LEN_ORIG, DEPTH, stride=1)
X_val, y_val = shuffle_initial(X_val, y_val, SEED)

# Memory freigeben
del X_val_raw, y_val_raw

# TENSORFLOW DATASETS
print("Erstelle TF Datasets...")
BATCH_SIZE = 8

def prepare_25d_input(x, y):
    # (D, H, W, 1) -> Squeeze -> (D, H, W) -> Transpose -> (H, W, D)
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) 
    
    # Y Center finden: (D, H, W, 1) -> Slice Mitte -> (H, W, 1)
    idx = tf.shape(y)[0] // 2 
    y_center = y[idx]
    return x, y_center

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
            .map(cast_to_float32, num_parallel_calls=AUTOTUNE)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(cast_to_float32, num_parallel_calls=AUTOTUNE)
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))


# training
BASE_NAME = f"unet_25d_D{DEPTH}_VarStride1-{MAX_STRIDE}"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME  = f"{BASE_NAME}__{RUN_ID}"

TB_ROOT     = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR  = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    make_epoch_ckpt_callback(RUN_NAME),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)

# Input Shape für Model definieren: (H, W, DEPTH)
model = unet_2d_stacked(input_shape=(192, 240, DEPTH))
model.compile(
    optimizer=optimizer,
    loss=mae_ssim_2d,
    metrics=[mae_center, ssim_center, psnr_center]
)

print(f"Starte Training: {RUN_NAME}")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=200,
    callbacks=callbacks,
    verbose=2
)

# Meta speichern
meta = make_meta_dict(
    script_name=RUN_NAME,
    batch_size=BATCH_SIZE,
    epochs=200,
    optimizer=optimizer,
    learning_rate=5e-4,
    input_shape=(192, 240, DEPTH),
    extra={"strides": f"1-{MAX_STRIDE}", "depth": DEPTH}
)

finalize_run(model, history, RUN_NAME, meta)
print("Fertig.")