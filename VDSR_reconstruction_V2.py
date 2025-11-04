# VDSR_reconstruction_V2.py
# ==============================
# 0) Imports & global setup
# ==============================
#!/usr/bin/env python3

# import os
import tensorflow as tf
from pathlib import Path
import h5py
import numpy as np

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

from VDSR_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict

# %%
# Jens original Code
def VDSR(input_shape, filters=64, kernel_initializer='he_normal'):
    """VDSR model architecture (Very Deep Super-Resolution Neural Network).

    - 'he_normal' weights initializer
    - 64 filters per layer
    - 20 convolutional layers
    - parametric rectifying linear unit (PReLU) as activation

    Reference: 
    J. Kim, J. K. Lee, and K. M. Lee, 
    “Accurate Image Super-Resolution Using Very Deep Convolutional Networks,” 
    in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2016, pp. 1646–1654. 
    doi: 10.1109/CVPR.2016.182.

    Parameters
    ----------
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    filters : int
        Number of filters per layer
    kernel_initializer : string
        Kernel initializer to be used as defined by keras.initializers
    
    Returns
    -------
    keras.Model
    """

    # Initialize a parametric linear rectifier unit
    para_relu = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.constant(0.25))

    # Create the neural network
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, activation=para_relu, kernel_initializer=kernel_initializer, padding='same') (input)

    for _ in range(19):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer=kernel_initializer, padding='same') (x)
        x = tf.keras.layers.Activation(para_relu) (x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=kernel_initializer, padding='same') (x)
    model = tf.keras.Model(input, x, name="VDSR")

    return model


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

def shuffle_initial(X, y, seed=None):
    """
    Shuffelt X und y mit der gleichen Permutation
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = np.arange(N) # Array mit Inizes[0, 1, 2, ..., N-1]
    rng.shuffle(indices)
    return X[indices], y[indices]   # Arrays nach Permutation neu anordnen


def augment_and_normalize_2d(scale_min: float, scale_max: float, p: float = 0.5):
    """
    Macht pro Sample in der Pipeline:
      1) Horizontal-Flip mit probability p
      2) Clipping auf [0, ∞)
      3) Normierung pro Bild durch Summe über Bild (H,W,C=1)
      4) zufällige Skalierung in [scale_min, scale_max]
      5) Clipping auf [0, 1]
    Erwartet x,y im Format (H, W, C) mit dtype float32.
    """
    def map_picture(x, y):
        # Flip, hier ist W Achse 1 (0:H, 1:W, 2:C) und die flippen wir
        flip = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[1]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[1]), lambda: y)

        # Clipping auf [0, ∞)
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # Normierung pro Bild (Summe über H,W,C)
        sum_x = tf.reduce_sum(x, axis=[0, 1, 2], keepdims=True) + 1e-12 # (1,1,1) bleibt (1,1,1)
        sum_y = tf.reduce_sum(y, axis=[0, 1, 2], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # Zufällige Skalierung pro Sample
        scale = tf.random.uniform(shape=[1, 1, 1],
                                  minval=tf.cast(scale_min, tf.float32),
                                  maxval=tf.cast(scale_max, tf.float32),
                                  dtype=tf.float32)
        x = x * scale
        y = y * scale

        # Clipping auf [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        return x, y

    return map_picture




# %%
# Daten einlesen
print("Lese Daten ein...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",}

RUN_NAME = "VDSR_reconstruction_V2"

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
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

# Batches hinzufügen
BATCH_SIZE = 8

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,amsgrad=True)
LOG_DIR = Path.home()/ "data" / "checkpoints_VDSR"
callbacks = [
    make_epoch_ckpt_callback(RUN_NAME),     # speichert nur das beste Modell in ~/data/checkpoints_VDSR
    tf.keras.callbacks.CSVLogger(str(LOG_DIR / f"{RUN_NAME}.csv"), append=True),
    tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{RUN_NAME}"),
]

# Compilieren
model = VDSR(input_shape=(192, 240, 1))
model.compile(
    optimizer=optimizer,
    loss='mae',       # MAE only
    metrics=['mae', 'mse', psnr_metric]
)


print("Erstelle Trainingsaten...")

AUTOTUNE = tf.data.AUTOTUNE

# Sicherstellen alles ist float 32
X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)

# Test: Mit Flip, Skala [5000, 15000]
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_2d(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

# Validation: kein Flip, quasi fixe Skala [10000, 10001]
val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_2d(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

print("Training beginnt...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
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