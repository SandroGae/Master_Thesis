# unet_25d_SSIM_middle_improved_V2.py
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

# Simples unet in 2.5D
POOL_HW = (1, 2, 2)  # (D, H, W) --> Kein Pooling über depth

def transformer_block(x, dim, num_heads, mlp_dim, dropout=0.1):
    # Layer Normalization 1 + Multi-Head Attention (Global Interaction)
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout)(x, x)
    x = layers.Add()([x, res]) # Residual Connection [cite: 673]

    # Layer Normalization 2 + Feed Forward (MLP)
    res = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dim)(x)
    x = layers.Dropout(dropout)(x)
    return layers.Add()([x, res])

def build_srdtrans(input_shape=(192, 240, 5), patch_size=4, embed_dim=128):
    inputs = layers.Input(shape=input_shape, name="input")

    # 1. Temporal Encoder [cite: 658, 660]
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(inputs)
    x = layers.ReLU()(x)

    # 2. Patch Embedding 
    h, w = input_shape[0], input_shape[1]
    num_patches = (h // patch_size) * (w // patch_size)
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(x) 
    curr_h, curr_w = x.shape[1], x.shape[2]
    
    # Flattening [cite: 665]
    x = layers.Reshape((num_patches, embed_dim))(x)
    
    # Learned Position Encoding 
    # In Keras functional muss das Embedding oft als Layer oder konstanter Tensor addiert werden
    pos_emb = tf.Variable(tf.zeros((1, num_patches, embed_dim)), trainable=True, name="pos_embedding")
    x = x + pos_emb

    # 3. Spatiotemporal Transformer Blocks (STB) [cite: 14, 76, 79]
    for _ in range(4):
        x = transformer_block(x, dim=embed_dim, num_heads=8, mlp_dim=embed_dim * 2)

    # 4. Temporal Decoder [cite: 672]
    x = layers.Reshape((curr_h, curr_w, embed_dim))(x)
    x = layers.Conv2DTranspose(embed_dim // 2, kernel_size=patch_size, strides=patch_size, padding="same")(x)
    x = layers.ReLU()(x)
    
    # Final Output (Rekonstruktion des mittleren Slices)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid", name="output")(x)

    return models.Model(inputs, outputs, name="srdtrans")



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




# Daten einlesen
print("Lade Daten...")

FILES = {   "training":   "/home/sgaell/data/original_data/training_data.hdf5",
            "validation": "/home/sgaell/data/original_data/validation_data.hdf5",}

BASE_NAME = "unet_25d_SSIM_middle_improved_V2"
RUN_ID    = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__bf{BASEFILTERS}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT    = Path.home() / "data" / "tblogs_unet_3d_simple"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Lade die Daten
X_train, y_train = load_split(FILES["training"])
X_val,   y_val   = load_split(FILES["validation"])

# Mache daraus Volumen im Format (N_vols = 2960, DEPTH, H=192, W=240, C=1)
X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val,   y_val   = make_sliding_windows(X_val,   y_val,   SERIES_LEN, DEPTH)

# Einmaliges initiales Shuffle (separat für Training und Validation):
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val,   y_val   = shuffle_initial(X_val,   y_val,   SEED)

# Batches hinzufügen
BATCH_SIZE = 8

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

# Compilieren
model = build_srdtrans(input_shape=(192, 240, DEPTH))

# Optimizer + callbacks
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, amsgrad=True)

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

def prepare_25d_input(x, y):
    """
    Wandelt 3D Input in 2D mit Channels um.
    Input x: (D, H, W, 1)  --> Output x: (H, W, D)
    Input y: (D, H, W, 1)  --> Output y: (H, W, 1) (Mitte)
    """
    # X reshape: Squeeze den letzten Channel (1), dann Transpose D nach hinten
    x = tf.squeeze(x, axis=-1)       # (D, H, W)
    x = tf.transpose(x, [1, 2, 0])   # (H, W, D) -> D ist jetzt Channel!

    # Y slice: Nur die Mitte holen
    depth = tf.shape(y)[0]
    idx = depth // 2
    y_center = y[idx]                # (H, W, 1) -> Slice nimmt eine Dimension weg, wir bleiben bei 3D Tensor
    
    return x, y_center

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=tf.data.AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE)
          .cache()
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))


print("Training beginnt...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    verbose=2
)

# Meta bauen
meta = make_meta_dict(
    script_name=RUN_NAME,
    batch_size=8,
    epochs=100,
    optimizer=optimizer,
    learning_rate=5e-4,
    input_shape=(192, 240, DEPTH),  
    scale_range_train=(5000,15000),
    scale_range_val=(10000,10001),
    extra={"loss": "mae_ssim(alpha=0.6)", "metrics": ["mae_center","mse_center","psnr_center","ssim_center"]}
)

final_path = finalize_run(model, history, RUN_NAME, meta)
print("Training beendet...")