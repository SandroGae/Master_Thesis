# VDSR_reconstruction_V2.py
# ==============================
# 0) Imports & global setup
# ==============================
#!/usr/bin/env python3

# import os
# XLA vor dem Import von TensorFlow abschalten
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
import tensorflow as tf
# tf.config.optimizer.set_jit(False)  # XLA JIT aus

from pathlib import Path
import h5py
import numpy as np

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
X_train, y_train = shuffle_initial(X_train, y_train, seed=0)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   seed=1)

# Data Augmentation auf Train und Val (Nur horizontal flip, mit p=0.5):
flipped_train = random_lr_flip(X_train, y_train, p=0.5, seed=0)
flipped_val   = random_lr_flip(X_val,   y_val,   p=0.5, seed=1)

# Normalisierung 
X_train, y_train = normalization(X_train, y_train, seed=0, scale_range=(5000,15000))
X_val,   y_val   = normalization(X_val,   y_val,   seed=1, scale_range=(10000,10001))

# Batches hinzufügen
BATCH_SIZE = 8

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,amsgrad=True)
callbacks = [
    make_epoch_ckpt_callback(RUN_NAME),     # speichert alle Epochen nach TEMPORARY
    tf.keras.callbacks.CSVLogger(f"{RUN_NAME}_history.csv", append=True),
    tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{RUN_NAME}")
]

# Compilieren
model = VDSR(input_shape=(192, 240, 1))
model.compile(
    optimizer=optimizer,
    loss='mae',       # MAE only
    metrics=['mae', 'mse', psnr_metric]
)

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

