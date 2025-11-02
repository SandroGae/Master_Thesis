# unet_2d_simple.py
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

from unet_2d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict

# %%
# Simples unet
POOL_HW = (2, 2)  # (H, W)

def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    x = layers.Conv2D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True)(x)
    x = layers.ReLU()(x)
    return x

def unet_2d(input_shape=(192, 240, 1), base_filters=16, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block_2d(inputs, base_filters)              ; p1 = layers.MaxPooling2D(POOL_HW)(c1)
    c2 = conv_block_2d(p1, base_filters * 2)              ; p2 = layers.MaxPooling2D(POOL_HW)(c2)
    c3 = conv_block_2d(p2, base_filters * 4)              ; p3 = layers.MaxPooling2D(POOL_HW)(c3)
    c4 = conv_block_2d(p3, base_filters * 8)              ; p4 = layers.MaxPooling2D(POOL_HW)(c4)

    # Bottleneck
    bn = conv_block_2d(p4, base_filters * 16)

    # Decoder
    u4 = layers.Conv2DTranspose(base_filters * 8, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])                   ; c5 = conv_block_2d(u4, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])                   ; c6 = conv_block_2d(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])                   ; c7 = conv_block_2d(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, kernel_size=POOL_HW, strides=POOL_HW, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])                   ; c8 = conv_block_2d(u1, base_filters)

    # Output: 1 Kanal, Sigmoid (0..1)
    out = layers.Conv2D(1, (1, 1), activation=output_activation,
                        kernel_initializer="he_normal", use_bias=True, name="output")(c8)

    return models.Model(inputs, out, name="unet_2d_simple_relu_sigmoid")



# Metriken
def psnr_metric(y_true, y_pred):
    # y_true, y_pred in [0, 1]
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))



def load_split(h5_path):
    """
    Lädt Daten aus HDF5-Datei und formatiert sie passend für VDSR
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

def shuffle_initial(X, y, seed):
    """
    Shuffelt X und y mit der gleichen Permutation
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = np.arange(N) # Array mit Inizes[0, 1, 2, ..., N-1]
    rng.shuffle(indices)
    return X[indices], y[indices]   # Arrays nach Permutation neu anordnen

def random_lr_flip(X, y, p=0.5, seed=None):
    """
    Links rechts FLip mit Wahrscheinlichkeit p
    """
    rng = np.random.default_rng(seed) # Zufallsgenerator mit Seed
    N = len(X)
    indices = np.arange(N) # Array mit Inizes[0, 1, 2, ..., N-1]
    random_values = rng.random(N) # uniform in [0, 1)
    flip_indices = indices[random_values < p]
    X[flip_indices] = X[flip_indices, :, ::-1, :] # Reverse pixelreihenfolge bei Width (N, H, W, C)
    y[flip_indices] = y[flip_indices, :, ::-1, :]


def normalization(X, y, seed=None, scale_range=None):
    """
    Normalisiert LC- und HC-Bilder analog wie Jens:
    1) Clipping auf [0, ∞)
    2) Normierung jedes Bildes durch seine Summe
    3) Multiplikation mit einem zufälligen Faktor ∈ [5000, 15000] für tain und val!
    4) Clipping auf [0, 1]

    Parameter
    ----------
    X, y : Arrays der Form (N, H, W, C=1)
    seed : Zufalls-Seed für Reproduzierbarkeit
    """
    X = np.clip(X, 0, None)
    y = np.clip(y, 0, None)

    # Durch Summe teilen (um Division durch 0 zu vermeiden: +1e-12)
    sum_X = np.sum(X, axis=(1, 2, 3), keepdims=True) + 1e-12 # Takes Height, Width und Channel für Summe (N, H, W, C)
    sum_y = np.sum(y, axis=(1, 2, 3), keepdims=True) + 1e-12
    X = X / sum_X
    y = y / sum_y

    # Zufälliger Faktor uniform aus [5000, 15000]
    N = len(X)
    rng = np.random.default_rng(seed)
    scale = rng.uniform(scale_range[0], scale_range[1], size=(N,1,1,1)).astype(np.float32)
    X_scaled = X * scale
    y_scaled = y * scale

    # Finales Clipping auf [0, 1]
    X = np.clip(X_scaled, 0, 1)
    y = np.clip(y_scaled, 0, 1)

    return X.astype(np.float32), y.astype(np.float32)



# %%
# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",}

RUN_NAME = "unet_2d_simple"

# Lade die Daten
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

# Check Formatierung
# print("TRAIN  X:", X_train.shape, X_train.dtype)  # (3280, 192, 240, 1) float32
# print("TRAIN  y:", y_train.shape, y_train.dtype)  # (3280, 192, 240, 1) float32
# print("VAL    X:", X_val.shape,   X_val.dtype)    # (820, 192, 240, 1) float32
# print("VAL    y:", y_val.shape,   y_val.dtype)    # (820, 192, 240, 1) float32


# %%
# Einmaliges initiales Shuffle (separat für Training und Validation):
X_train, y_train = shuffle_initial(X_train, y_train, seed=0)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   seed=1)

# Data Augmentation auf Train und Val (Nur horizontal flip, mit p=0.5):
random_lr_flip(X_train, y_train, p=0.5, seed=0)
random_lr_flip(X_val,   y_val,   p=0.5, seed=1)

# Normalisierung 
X_train, y_train = normalization(X_train, y_train, seed=0, scale_range=(5000,15000))
X_val,   y_val   = normalization(X_val,   y_val,   seed=1, scale_range=(10000,10001))

# Batches hinzufügen
BATCH_SIZE = 8

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,amsgrad=True)
LOG_DIR = Path.home()/ "data" / "checkpoints_unet_2d_simple"
callbacks = [
    make_epoch_ckpt_callback(RUN_NAME),     # speichert nur das beste Modell in ~/data/checkpoints_unet_2d_simple
    tf.keras.callbacks.CSVLogger(str(LOG_DIR / f"{RUN_NAME}.csv"), append=True),
    tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{RUN_NAME}"),
]

# Compilieren
model = unet_2d(input_shape=(192, 240, 1))
model.compile(
    optimizer=optimizer,
    loss='mae',       # MAE only
    metrics=['mae', 'mse', psnr_metric]
)

print("Training beginnt...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=100,
    shuffle=True, # Shuffel intern pro Epoche
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
    input_shape=(192,240,1),
    scale_range_train=(5000,15000),
    scale_range_val=(10000,10001),
    extra={"loss": "mae", "metrics": ["mae", "mse", "psnr"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)

print("Training beendet...")