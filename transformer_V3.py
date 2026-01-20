# transformer_V3
#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Optional: gleiche Helper wie im UNet-Skript (falls bei dir vorhanden)
try:
    from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
    from tb_utils import tb_callbacks
    _HAS_HELPERS = True
except Exception:
    _HAS_HELPERS = False


# -----------------------------
# Reproduzierbarkeit
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# (wie im UNet-Skript)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# Optional: GPU memory growth (hilft oft bei Fragmentierung)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


# -----------------------------
# Parameter
# -----------------------------
DEPTH = 5
SERIES_LEN = 41

EMBED_DIM = 96
WINDOW_SIZE = 8

BATCH_SIZE = 8
EPOCHS = 100

FILES = {
    "training": "/home/sgaell/data/original_data/training_data.hdf5",
    "validation": "/home/sgaell/data/original_data/validation_data.hdf5",
}


# -----------------------------
# Daten: exakt wie UNet (sliding windows + augment+normalize)
# -----------------------------
def load_split(h5_path: str):
    """
    Lädt Daten aus HDF5-Datei:
      low_count/data, high_count/data: (H, W, N)
    Rückgabe:
      low_count, high_count: (N, H, W, 1)
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]      # (H, W, N)
        high_count = f["high_count/data"][:]    # (H, W, N)

    low_count = np.moveaxis(low_count, -1, 0)    # (N, H, W)
    high_count = np.moveaxis(high_count, -1, 0)

    low_count = low_count[:, :, :, np.newaxis]   # (N, H, W, 1)
    high_count = high_count[:, :, :, np.newaxis] # (N, H, W, 1)

    return low_count, high_count


def make_sliding_windows(X, y, series_len=None, depth=None):
    """
    X, y: (N, H, W, 1)
    return: (N_vols, D, H, W, 1)
    """
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1

    X_volumes = []
    y_volumes = []

    for i in range(0, n_series, 1):
        start = i * series_len
        blockX = X[start:start + series_len]
        blockY = y[start:start + series_len]

        for start_idx in range(0, n_vols_per_series, 1):
            X_volumes.append(blockX[start_idx:start_idx + depth])
            y_volumes.append(blockY[start_idx:start_idx + depth])

    X_volumes = np.stack(X_volumes, axis=0)
    y_volumes = np.stack(y_volumes, axis=0)
    return X_volumes, y_volumes


def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]


def augment_and_normalize_3d_per_slice(scale_min: float, scale_max: float, p: float = 0.5):
    """
    Erwartet x,y je Sample als (D, H, W, 1) float32.
    Schritte je Sample (wie UNet):
      1) Horizontal-Flip (W-Achse) mit probability p, für alle Slices identisch
      2) Clip >= 0
      3) Norm pro Slice: Division durch Summe über (H,W,C) -> (D,1,1,1)
      4) Zufalls-Skalierung je Sample: gleiche Skala alle Slices
    """
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32) < tf.constant(p, tf.float32)
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)  # axis=2 ist W
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)

        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        scale = tf.random.uniform([], minval=scale_min, maxval=scale_max, dtype=tf.float32)
        x = x * scale
        y = y * scale
        return x, y

    return map_volume


def prepare_transformer_input(x, y):
    """
    Wie prepare_25d_input im UNet, aber Output passend fürs Transformer-Modell:
      Input x: (D, H, W, 1)  -> Output x: (H, W, D)
      Input y: (D, H, W, 1)  -> Output y: (H, W, 1) (Mitte)
    """
    x = tf.squeeze(x, axis=-1)       # (D, H, W)
    x = tf.transpose(x, [1, 2, 0])   # (H, W, D)

    depth = tf.shape(y)[0]
    idx = depth // 2
    y_center = y[idx]               # (H, W, 1)

    return x, y_center


# -----------------------------
# Loss + Metriken: exakt wie UNet
# -----------------------------
def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_mean = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_mean)


def mae_center(y_true, y_pred):
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
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)


def ssim_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


# -----------------------------
# Swin-Helpers
# -----------------------------
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, window_size, window_size, C))


def window_reverse(windows, window_size, h, w):
    c = windows.shape[-1]
    x = tf.reshape(windows, (-1, h // window_size, w // window_size, window_size, window_size, c))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, h, w, c))


class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size, shift_size=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(dim * 4, activation=tf.nn.gelu),
            layers.Dense(dim),
        ])

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        res = x
        x = self.norm1(x)

        if self.shift_size > 0:
            # Einfacher Shift (wie in deinem Code)
            x = tf.pad(x, [[0, 0], [self.shift_size, 0], [self.shift_size, 0], [0, 0]])
            x = x[:, :h, :w, :]

        x_windows = window_partition(x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, self.dim))

        attn_windows = self.attn(x_windows, x_windows)
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, self.dim))

        x = window_reverse(attn_windows, self.window_size, h, w)
        x = layers.Add()([res, x])

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return layers.Add()([res, x])


class LearnedPositionalEncoding(layers.Layer):
    def __init__(self, seq_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = int(seq_length)
        self.embedding_dim = int(embedding_dim)
        self.pos_embeddings = self.add_weight(
            name="pos_embedding",
            shape=(1, self.seq_length, self.embedding_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
        })
        return config



# -----------------------------
# Modell
# -----------------------------
def build_srdtrans_swin(input_shape=(192, 240, DEPTH), embed_dim=EMBED_DIM):
    inputs = layers.Input(shape=input_shape, name="input")  # (H, W, D)
    h, w, d = input_shape
    p = h * w

    # =========================
    # 1) TEMPORAL TRANSFORMER (pro Pixel über D)
    # =========================
    # (B, H, W, D) -> (B, P, D, 1)
    xt = layers.Reshape((p, d, 1))(inputs)

    # (B, P, D, 1) -> (B*P, D, 1)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, d, 1)))(xt)

    # Projektion auf 4 Features
    xt = layers.Dense(4, name="temp_proj")(xt)  # (B*P, D, 4)

    # Positional Encoding über D
    xt = LearnedPositionalEncoding(seq_length=d, embedding_dim=4)(xt)

    for _ in range(2):
        res_t = xt
        xt = layers.LayerNormalization(epsilon=1e-6)(xt)
        xt = layers.MultiHeadAttention(num_heads=2, key_dim=2)(xt, xt)  # 2*2 = 4
        xt = layers.Add()([res_t, xt])

        res_t = xt
        xt = layers.LayerNormalization(epsilon=1e-6)(xt)
        xt = layers.Dense(8, activation="gelu")(xt)
        xt = layers.Dense(4)(xt)
        xt = layers.Add()([res_t, xt])

    # zurück: (B*P, D, 4) -> (B, P, D, 4)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, p, d, 4)))(xt)

    # (B, P, D, 4) -> (B, H, W, D, 4)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, h, w, d, 4)))(xt)

    # (B, H, W, D, 4) -> (B, H, W, D*4)
    xt = layers.Reshape((h, w, d * 4))(xt)

    # Projektion auf D Kanäle via 1x1 Conv
    xt = layers.Conv2D(d, kernel_size=1, padding="same", name="temp_out")(xt)  # (B, H, W, D)

    # =========================
    # 2) SPATIAL SWIN
    # =========================
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(xt)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=0)(x)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=WINDOW_SIZE // 2)(x)

    # =========================
    # 3) DECODER
    # =========================
    x = layers.Conv2D(embed_dim // 2, kernel_size=3, padding="same")(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid", name="output")(x)

    return models.Model(inputs, outputs, name="srdtrans_swin")


# -----------------------------
# RUN
# -----------------------------
print("Lade Daten...")

X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])

# Sliding windows wie im UNet
X_train, y_train = make_sliding_windows(X_train, y_train, SERIES_LEN, DEPTH)
X_val, y_val = make_sliding_windows(X_val, y_val, SERIES_LEN, DEPTH)

# Einmaliges initiales Shuffle (wie im UNet)
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val, y_val = shuffle_initial(X_val, y_val, SEED)

# Float32 wie im UNet
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True)
    .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=AUTOTUNE)
    .map(prepare_transformer_input, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=AUTOTUNE)
    .map(prepare_transformer_input, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Modell
model = build_srdtrans_swin(input_shape=(192, 240, DEPTH), embed_dim=EMBED_DIM)

# Optimizer wie im UNet
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)

# Compile: Loss + Metriken wie im UNet
model.compile(
    optimizer=optimizer,
    loss=mae_ssim_2d,
    metrics=[mae_center, mse_center, psnr_center, ssim_center],
)

# Run-Namen / Callbacks (optional gleich wie UNet)
BASE_NAME = "srdtrans_swin_improved_pipeline"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__emb{EMBED_DIM}__D{DEPTH}__lossMAE_SSIM__{RUN_ID}"

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=2),
    tf.keras.callbacks.CSVLogger(str(TB_RUN_DIR / f"{RUN_NAME}.csv"), append=False),
]

if _HAS_HELPERS:
    callbacks.append(make_epoch_ckpt_callback(RUN_NAME))
    callbacks.extend(tb_callbacks(TB_RUN_DIR))
else:
    # Fallback: bestes Modell speichern
    ckpt_dir = Path("checkpoints") / RUN_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
    )

print("Training beginnt...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

if _HAS_HELPERS:
    meta = make_meta_dict(
        script_name=RUN_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        optimizer=optimizer,
        learning_rate=5e-4,
        input_shape=(192, 240, DEPTH),
        scale_range_train=(5000, 15000),
        scale_range_val=(10000, 10001),
        extra={
            "loss": "mae_ssim(alpha=0.6)",
            "metrics": ["mae_center", "mse_center", "psnr_center", "ssim_center"],
            "model": "srdtrans_swin",
            "embed_dim": EMBED_DIM,
            "window_size": WINDOW_SIZE,
        },
    )
    final_path = finalize_run(model, history, RUN_NAME, meta)
    print(f"Training beendet. Final gespeichert: {final_path}")
else:
    print("Training beendet.")
