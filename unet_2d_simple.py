# unet_2d_simple.py

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

from unet_2d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks, ImageLogger


# %%
# Simples unet in 2d
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

    # Output Sigmoid
    out = layers.Conv2D(1, (1, 1), activation=output_activation,
                        kernel_initializer="he_normal", use_bias=True, name="output")(c8)

    return models.Model(inputs, out, name="unet_2d_simple_relu_sigmoid")



# Metriken
def psnr_metric(y_true, y_pred):
    # y_true, y_pred in [0, 1]
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))



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



def shuffle_initial(X, y, seed):
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
        scale = tf.random.uniform(shape=[1, 1, 1], minval=tf.cast(scale_min, tf.float32), maxval=tf.cast(scale_max, tf.float32), dtype=tf.float32)
        x = x * scale
        y = y * scale

        # Clipping auf [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        return x, y

    return map_picture



# %%
# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
            "test":       "/home/sgaell/data/original_data/test_data.hdf5",}

BASE_NAME = "unet_2d_simple"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME  = f"{BASE_NAME}__seed{SEED}__bf{16}__lossMAE__{RUN_ID}"

# Tensorboard run Verzeichnis
TB_ROOT    = Path.home() / "data" / "tblogs_unet_2d_simple"
TB_RUN_DIR = make_run_dir(RUN_NAME, root=TB_ROOT)

# Lade die Daten
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])
X_test, y_test = load_split(FILES["test"])

# %%
# Einmaliges initiales Shuffle (separat für Training und Validation):
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)



# Batches hinzufügen
BATCH_SIZE = 8

print("Erstelle Trainingsaten...")

AUTOTUNE = tf.data.AUTOTUNE

# Sicherstellen alles ist float 32
X_train = X_train.astype(np.float32); y_train = y_train.astype(np.float32)
X_val   = X_val.astype(np.float32);   y_val   = y_val.astype(np.float32)
X_test = X_test.astype(np.float32); y_test = y_test.astype(np.float32)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_2d(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_2d(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

# Tensorboard Bilder loggen
idx = 468
x_vis_test, y_vis_test = next(iter(
    tf.data.Dataset.from_tensor_slices((X_test[idx:idx+1], y_test[idx:idx+1]))
    .map(augment_and_normalize_2d(10000.0, 10001.0, p=0.0))
    .batch(1)
))

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,amsgrad=True)

callbacks = [
    make_epoch_ckpt_callback(RUN_NAME),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR, histograms=False, profile=False),
    ImageLogger(TB_RUN_DIR, (x_vis_test, y_vis_test), every_n_epochs=1, max_outputs=1),  # Bilder-Reiter
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
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    verbose=2
)

# Meta bauen
meta = make_meta_dict(
    script_name=RUN_NAME,
    batch_size=8,
    epochs=5,
    optimizer=optimizer,
    learning_rate=5e-4,
    input_shape=(192,240,1),
    scale_range_train=(5000,15000),
    scale_range_val=(10000,10001),
    extra={"loss": "mae", "metrics": ["mae", "mse", "psnr"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)

print("Training beendet...")