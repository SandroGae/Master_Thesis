# unet_3d_simple.py
# ==============================
# 0) Imports & global setup
# ==============================
#!/usr/bin/env python3

# import os
# XLA vor dem Import von TensorFlow abschalten
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
import tensorflow as tf
# tf.config.optimizer.set_jit(False)  # XLA JIT aus
from tensorflow.keras import layers, models
from pathlib import Path
import h5py
import numpy as np

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict

# %%
# Simples unet in 3d
POOL_HW = (1, 2, 2)  # (D, H, W) --> Kein Pooling über depth

def conv_block_3d(x, filters, kernel_size=(1, 3, 3), padding="same"):
    ki = "he_normal"
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True)(x)
    x = layers.ReLU()(x)
    return x

def unet_3d(input_shape=(5, 192, 240, 1), base_filters=16, output_activation="sigmoid"):
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
    out = layers.Conv3D(1, (1, 1, 1), activation=output_activation,
                        kernel_initializer="he_normal", use_bias=True, name="output")(c8)

    return models.Model(inputs, out, name="unet_3d_simple_relu_sigmoid")



# Metrik: PSNR pro Sample (Volumen)
def psnr_metric_3d_per_sample(y_true, y_pred):
    # y_*: (N, D, H, W, C) in [0,1]
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3,4))  # (N,)
    psnr = 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)               # (N,)
    return psnr  # Keras zeigt dir den Mittelwert über N an



def load_split(h5_path):
    """
    Lädt Daten aus HDF5-Datei und formatiert sie passend für 2d unet
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]      # (H, W, N)
        high_count = f["high_count/data"][:]    # (H, W, N)

    # Achse verschieben: (H, W, N) -> (N, H, W)
    low_count = np.moveaxis(low_count, -1, 0) # Nimmt letzte Achse un schiebt sie an den Anfang
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
    n_series = N // series_len              # Anzahl Serien in den Daten
    n_vols_per_series = series_len - depth + 1     # Anzahl Bilder pro Serie

    X_volumes = []
    y_volumes = []

    for i in range(0, n_series, 1):
        start = i * series_len   # Start der spezifischen Bilderserie
        blockX = X[start:start+series_len]  # (41,H,W,C=1)
        blockY = y[start:start+series_len]  # (41,H,W,C=1)

        for start_idx in range(0, n_vols_per_series, 1):
            X_volumes.append(blockX[start_idx : start_idx + depth])  # Liste aus 37 Volumen mit Dimension (5,192,240,C=1) pro Durchlauf bis zu 80 Serien * 37 Volumes = 2960 Elementen
            y_volumes.append(blockY[start_idx : start_idx + depth])

    X_volumes = np.stack(X_volumes, axis=0) # [(5,192,240,C=1), (5,192,240,C=1), (5,192,240,C=1), ... , (5,192,240,C=1)] --> (N_vols = 2960, 5, 192, 240, 1)
    y_volumes = np.stack(y_volumes, axis=0)
    return X_volumes, y_volumes



def shuffle_initial(X, y, seed):
    """
    Shuffelt X und y mit der gleichen Permutation
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = np.arange(N) # Array mit Inizes[0, 1, 2, ..., N-1]
    rng.shuffle(indices)
    return X[indices], y[indices]   # Arrays nach Permutation neu anordnen


def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    """
    Erwartet x,y je Sample als (D, H, W, C) float32.
    Schritte je Sample:
      1) Horizontal-Flip (W-Achse) mit probability p, für alle Slices identisch
      2) Clip >= 0
      3) Norm pro Slice: Division durch Summe über (H,W,C) -> Form (D,1,1,1)
      4) Zufalls-Skalierung je Sample: eigene Skala pro Slice (Form (D,1,1,1))
      5) Clip auf [0,1]
    """
    def _map(x, y):
        # Flip über Breite (axis=2 bei (D,H,W,C)? Vorsicht: deine Reihenfolge ist (D,H,W,C)
        # Nun ist W die Achse 2 (0:D, 1:H, 2:W, 3:C).
        prob = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32)
        flip_mask = tf.less(prob, tf.cast(p, tf.float32))
        def _flip(t): 
            return tf.reverse(t, axis=[2])  # axis=2 = Width bei (D,H,W,C)
        x = tf.cond(flip_mask, lambda: _flip(x), lambda: x)
        y = tf.cond(flip_mask, lambda: _flip(y), lambda: y)

        # Clip >= 0
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # Normierung pro Slice: Summe ueber (H,W,C) -> Achsen [1,2,3].
        sum_x = tf.reduce_sum(x, axis=[1,2,3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1,2,3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # Zufalls-Skalierung + Clip auf [0,1]
        D = tf.shape(x)[0]  # Depth
        scale = tf.random.uniform([D,1,1,1], minval=tf.cast(scale_min, tf.float32), maxval=tf.cast(scale_max, tf.float32), dtype=tf.float32)
        x = tf.clip_by_value(x * scale, 0.0, 1.0)
        y = tf.clip_by_value(y * scale, 0.0, 1.0)

        return x, y
    return _map



# %%
# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",}

RUN_NAME = "unet_3d_simple"

# Lade die Daten
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

# Check Formatierung
# print("TRAIN  X:", X_train.shape, X_train.dtype)  # (3280, 192, 240, 1) float32
# print("TRAIN  y:", y_train.shape, y_train.dtype)  # (3280, 192, 240, 1) float32
# print("VAL    X:", X_val.shape,   X_val.dtype)    # (820, 192, 240, 1) float32
# print("VAL    y:", y_val.shape,   y_val.dtype)    # (820, 192, 240, 1) float32

# Mache daraus Volumen im Format (N_vols = 2960, D=5, H=192, W=240, C=1)
DEPTH = 5
SERIES_LEN = 41
X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val,   y_val,   SERIES_LEN, DEPTH)

# %%
# Einmaliges initiales Shuffle (separat für Training und Validation):
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

# Batches hinzufügen
BATCH_SIZE = 8

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,amsgrad=True)
LOG_DIR = Path.home()/ "data" / "checkpoints_unet_3d_simple"
callbacks = [
    make_epoch_ckpt_callback(RUN_NAME),     # speichert nur das beste Modell in ~/data/checkpoints_unet_3d_simple
    tf.keras.callbacks.CSVLogger(str(LOG_DIR / f"{RUN_NAME}.csv"), append=True),
    tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{RUN_NAME}"),
]

# Compilieren
model = unet_3d(input_shape=(5, 192, 240, 1))
model.compile(
    optimizer=optimizer,
    loss='mae',       # MAE only
    metrics=['mae', 'mse', psnr_metric_3d_per_sample]
)

print("Erstelle Trainingsdaten…")

AUTOTUNE = tf.data.AUTOTUNE

# Sicherstellen alles ist float 32
X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

print("Training beginnt...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)

# Meta bauen
meta = make_meta_dict(
    script_name=RUN_NAME,
    batch_size=8,
    epochs=100,
    optimizer=optimizer,
    learning_rate=5e-4,
    input_shape=(5, 192, 240, 1),  # 3D-Input
    scale_range_train=(5000,15000),
    scale_range_val=(10000,10001),
    extra={"loss": "mae", "metrics": ["mae", "mse", "psnr_metric_3d_per_sample"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)

print("Training beendet...")