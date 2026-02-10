#!/usr/bin/env python3

import os
import random
import h5py
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from tensorflow.keras import layers, models

# Deine Helper-Skripte (wie in V1 & UNet)
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# --- REPRODUZIERBARKEIT ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# --- PARAMETER ---
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

# --- SWIN-HELPERS ---

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
            layers.Dense(dim)
        ])

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        res = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = tf.pad(x, [[0, 0], [self.shift_size, 0], [self.shift_size, 0], [0, 0]])
            x = x[:, :h, :w, :]
        x_windows = window_partition(x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, self.dim))
        attn_windows = self.attn(x_windows, x_windows)
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, self.dim))
        x = window_reverse(attn_windows, self.window_size, h, w)
        x = layers.Add()([res, x])
        res = x
        x = self.norm2(x)
        return layers.Add()([res, self.mlp(x)])

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

# --- MODELL (SRDTrans Swin Architecture) ---

def build_srdtrans_swin(input_shape=(192, 240, 5), embed_dim=96):
    inputs = layers.Input(shape=input_shape)
    h, w, d = input_shape
    p = h * w

    # 1) TEMPORAL TRANSFORMER
    xt = layers.Reshape((p, d, 1))(inputs)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, d, 1)))(xt)
    xt = layers.Dense(4, name="temp_proj")(xt)
    xt = LearnedPositionalEncoding(seq_length=d, embedding_dim=4)(xt)

    for _ in range(2):
        res_t = xt
        xt = layers.LayerNormalization(epsilon=1e-6)(xt)
        xt = layers.MultiHeadAttention(num_heads=2, key_dim=2)(xt, xt)
        xt = layers.Add()([res_t, xt])
        res_t = xt
        xt = layers.LayerNormalization(epsilon=1e-6)(xt)
        xt = layers.Dense(8, activation="gelu")(xt)
        xt = layers.Dense(4)(xt)
        xt = layers.Add()([res_t, xt])

    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, p, d, 4)))(xt)
    xt = layers.Lambda(lambda t: tf.reshape(t, (-1, h, w, d, 4)))(xt)
    xt = layers.Reshape((h, w, d * 4))(xt)
    xt = layers.Conv2D(d, kernel_size=1, padding="same", name="temp_out")(xt)

    # 2) SPATIAL SWIN
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(xt)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=0)(x)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=WINDOW_SIZE // 2)(x)

    # 3) DECODER
    x = layers.Conv2D(embed_dim // 2, kernel_size=3, padding="same")(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="srdtrans_swin_v2")

# --- METRIKEN (Exakt wie UNet) ---

def mae_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true, y_pred = tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# --- DATA GENERATOR ---

class XRDDataGenerator:
    def __init__(self, h5_path, series_len, depth, is_train=True):
        self.h5_path = h5_path
        self.series_len = series_len
        self.depth = depth
        self.is_train = is_train
        with h5py.File(h5_path, "r") as f:
            self.total_samples = f["low_count/data"].shape[-1]
            self.n_series = self.total_samples // series_len
            self.vols_per_series = series_len - depth + 1
            self.indices = np.arange(self.n_series * self.vols_per_series)

    def __call__(self):
        with h5py.File(self.h5_path, "r") as f:
            low_data, high_data = f["low_count/data"], f["high_count/data"]
            if self.is_train:
                np.random.shuffle(self.indices)

            for idx in self.indices:
                s_idx, i_idx = idx // self.vols_per_series, idx % self.vols_per_series
                start = s_idx * self.series_len + i_idx
                x_vol = low_data[:, :, start : start + self.depth].astype(np.float32)
                y_vol = high_data[:, :, start : start + self.depth].astype(np.float32)

                for dd in range(self.depth):
                    x_vol[:, :, dd] /= (np.sum(x_vol[:, :, dd]) + 1e-12)
                    y_vol[:, :, dd] /= (np.sum(y_vol[:, :, dd]) + 1e-12)

                scale = random.uniform(5000, 15000) if self.is_train else 10000
                yield x_vol * scale, (y_vol[:, :, self.depth // 2] * scale)[:, :, np.newaxis]

# --- MAIN RUN ---

# Run-Namen Logik wie in V1
BASE_NAME = "transformer_V2"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{BASE_NAME}__seed{SEED}__emb{EMBED_DIM}__D{DEPTH}__lossMAE__{RUN_ID}"

TB_ROOT = Path.home() / "data" / "tblogs_transformer"
CKPT_FOLDER = "checkpoints_transformer"
TB_RUN_DIR = TB_ROOT / RUN_NAME
TB_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
output_sig = (
    tf.TensorSpec(shape=(192, 240, 5), dtype=tf.float32),
    tf.TensorSpec(shape=(192, 240, 1), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    XRDDataGenerator(FILES["training"], SERIES_LEN, DEPTH, True),
    output_signature=output_sig
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    XRDDataGenerator(FILES["validation"], SERIES_LEN, DEPTH, False),
    output_signature=output_sig
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Modell & Optimizer
model = build_srdtrans_swin(embed_dim=EMBED_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=True)

model.compile(
    optimizer=optimizer, 
    loss="mae", 
    metrics=[mae_center, mse_center, psnr_center, ssim_center]
)

# Callbacks (V1 Style)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=2),
    make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
    tf.keras.callbacks.CSVLogger(str(Path.home() / "data" / CKPT_FOLDER / f"{RUN_NAME}.csv"), append=False),
    *tb_callbacks(TB_RUN_DIR),
]

print(f"Starte Training: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# Finalisierung & Meta (V1 Style)
meta = make_meta_dict(
    script_name=RUN_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS, 
    optimizer=optimizer, learning_rate=1e-4, input_shape=(192, 240, DEPTH),
    extra={"loss": "MAE", "model": "Swin_Transformer_V2"}
)

finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
print("Training beendet.")