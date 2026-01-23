# unet_25d_MAE_MSE_SSIM.py
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

# Reproduzierbatkeit
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


# Parameters
DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64
BATCH_SIZE = 8
EPOCHS = 100
ALPHA_LIST = [0.0]
BETA_LIST = [1.0]
# ALPHA_LIST = np.linspace(0.0, 1.0, 7)
# BETA_LIST  = np.linspace(0.0, 1.0, 7)

# Simples unet in 2.5D
POOL_HW = (1, 2, 2)  # (D, H, W) --> Kein Pooling über depth

def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder (2D Pooling reduziert H und W)
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

    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8) # Direkt 1 Channel (kein Lambda Slicing mehr nötig)
    
    return models.Model(inputs, out, name="unet_25d_stacked")



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
        # W Achse ist 2 (0:D, 1:H, 2:W, 3:C), die flippen
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


def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 10
    base_lr = 5e-4
    if epoch < warmup_epochs:
        # Linearer Anstieg
        return base_lr * (epoch + 1) / warmup_epochs
    return lr

# Loss
def get_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Einzel-Metriken berechnen
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        # Kaskadierte Gewichtung
        w_ssim = alpha
        w_mse  = (1.0 - alpha) * beta
        w_mae  = (1.0 - alpha) * (1.0 - beta)

        return (w_ssim * ssim_loss) + (w_mse * mse) + (w_mae * mae)
    return loss


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

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0]) # (H, W, D)
    y_center = y[tf.shape(y)[0] // 2] # Mitte
    return x, y_center



# Daten einlesen
print("Lade Daten...")
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5",
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}

X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val, y_val = shuffle_initial(X_val, y_val, SEED)

X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32); y_val = y_val.astype(np.float32)

# Datasets einmalig erstellen
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# TRAINING SWEEP
for alpha_val in ALPHA_LIST:
    for beta_val in BETA_LIST: # Verschachtelte Schleife für Beta
        
        # Werte runden für saubere Pfadnamen
        a_r = round(float(alpha_val), 2)
        b_r = round(float(beta_val), 2)

        BASE_NAME = "unet_25d_TripleLoss"
        RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
        # RUN_NAME enthält jetzt beide Parameter
        RUN_NAME = f"{BASE_NAME}_a{a_r}_b{b_r}_bf{BASEFILTERS}_D{DEPTH}_{RUN_ID}"
        
        TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
        TB_RUN_DIR = TB_ROOT / RUN_NAME
        TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n" + "="*60)
        print(f"STARTE TRAINING: Alpha={a_r}, Beta={b_r}")
        print(f"Run-Name: {RUN_NAME}")
        print("="*60 + "\n")

        # Modell & Optimizer in jeder Runde neu instanziieren
        model = unet_2d_stacked()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
        
        current_callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
            make_epoch_ckpt_callback(RUN_NAME),
            tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"log_{RUN_NAME}.csv")),
            *tb_callbacks(TB_RUN_DIR)
        ]

        # Compilieren mit beiden Parametern
        model.compile(optimizer=optimizer, 
                      loss=get_triple_loss(alpha=a_r, beta=b_r), 
                      metrics=[mae_center, mse_center, psnr_center, ssim_center])

        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, 
                            callbacks=current_callbacks, verbose=2)

        # Meta-Daten mit allen Gewichtungsinformationen
        w_ssim = a_r
        w_mse  = (1.0 - a_r) * b_r
        w_mae  = (1.0 - a_r) * (1.0 - b_r)
        
        meta = make_meta_dict(
            script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
            optimizer=optimizer, learning_rate=5e-4, input_shape=(192, 240, DEPTH),
            extra={
                "alpha": a_r, "beta": b_r,
                "w_ssim": round(w_ssim, 4), "w_mse": round(w_mse, 4), "w_mae": round(w_mae, 4),
                "loss": f"triple_loss(a={a_r}, b={b_r})"
            }
        )

        finalize_run(model, history, RUN_NAME, meta)
        
        # --- SPEICHER-HYGIENE (WICHTIG!) ---
        # Leert den RAM und VRAM, damit die GPU nicht nach 5 Läufen voll ist
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

print("\n--- Das 7x7 Grid (49 Runs) wurde erfolgreich abgearbeitet! ---")