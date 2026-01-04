#!/usr/bin/env python3

import os
import random
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
from pathlib import Path

# --- REPRODUZIERBARKEIT & SETUP ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- PARAMETER ---
DEPTH = 5             
SERIES_LEN = 41
EMBED_DIM = 96        
WINDOW_SIZE = 8       
BATCH_SIZE = 8        
EPOCHS = 100

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
    """Swin-Block mit Padding statt Rolling für XRD-Datenkonsistenz."""
    def __init__(self, dim, num_heads, window_size, shift_size=0, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads, self.window_size, self.shift_size = dim, num_heads, window_size, shift_size
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
        self.pos_embeddings = self.add_weight(
            shape=(1, seq_length, embedding_dim), initializer="zeros", trainable=True)
    def call(self, x): return x + self.pos_embeddings

# --- MODELL-ARCHITEKTUR ---

def build_srdtrans_swin(input_shape=(192, 240, 5), embed_dim=96):
    inputs = layers.Input(shape=input_shape)
    h, w, d = input_shape

    # 1. Temporal Transformer (Pixel-wise)
    xt = layers.Reshape((h * w, d, 1))(inputs)
    xt = LearnedPositionalEncoding(seq_length=d, embedding_dim=1)(xt)
    for _ in range(2):
        res_t = xt
        xt = layers.LayerNormalization()(xt)
        xt = layers.MultiHeadAttention(num_heads=1, key_dim=4)(xt, xt)
        xt = layers.Add()([res_t, xt])
        res_t = xt
        xt = layers.LayerNormalization()(xt)
        xt = layers.Dense(4, activation="gelu")(xt); xt = layers.Dense(1)(xt)
        xt = layers.Add()([res_t, xt])
    xt = layers.Reshape((h, w, d))(xt)

    # 2. Spatial Swin Transformer
    x = layers.Conv2D(embed_dim, kernel_size=3, padding="same")(xt)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=0)(x)
    x = SwinTransformerBlock(dim=embed_dim, num_heads=8, window_size=WINDOW_SIZE, shift_size=WINDOW_SIZE // 2)(x)

    # 3. Decoder
    x = layers.Conv2D(embed_dim // 2, kernel_size=3, padding="same")(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")(x)
    return models.Model(inputs, outputs)

# --- EFFIZIENTER DATA GENERATOR (FIXED) ---

class XRDDataGenerator:
    def __init__(self, h5_path, series_len, depth, is_train=True):
        self.h5_path, self.series_len, self.depth, self.is_train = h5_path, series_len, depth, is_train
        with h5py.File(h5_path, "r") as f:
            self.total_samples = f["low_count/data"].shape[-1]
            self.n_series = self.total_samples // series_len
            self.vols_per_series = series_len - depth + 1
            self.indices = np.arange(self.n_series * self.vols_per_series)

    def __call__(self):
        with h5py.File(self.h5_path, "r") as f:
            low_data, high_data = f["low_count/data"], f["high_count/data"]
            if self.is_train: np.random.shuffle(self.indices)

            for idx in self.indices:
                s_idx, i_idx = idx // self.vols_per_series, idx % self.vols_per_series
                start = s_idx * self.series_len + i_idx
                
                x_vol = low_data[:, :, start : start + self.depth].astype(np.float32)
                y_vol = high_data[:, :, start : start + self.depth].astype(np.float32)

                # Augmentation & Normierung
                if self.is_train and random.random() < 0.5:
                    x_vol, y_vol = x_vol[:, ::-1, :], y_vol[:, ::-1, :]

                for d in range(self.depth):
                    sum_x = np.sum(x_vol[:,:,d]) + 1e-12
                    sum_y = np.sum(y_vol[:,:,d]) + 1e-12
                    x_vol[:,:,d] /= sum_x
                    y_vol[:,:,d] /= sum_y
                
                scale = random.uniform(5000, 15000) if self.is_train else 10000
                x_vol *= scale; y_vol *= scale

                yield x_vol, y_vol[:, :, self.depth // 2][:, :, np.newaxis]

# --- TRAINING SETUP ---

FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5", 
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}

output_sig = (tf.TensorSpec(shape=(192, 240, 5), dtype=tf.float32),
              tf.TensorSpec(shape=(192, 240, 1), dtype=tf.float32))

train_ds = tf.data.Dataset.from_generator(XRDDataGenerator(FILES["training"], SERIES_LEN, DEPTH, True), 
                                          output_signature=output_sig).batch(BATCH_SIZE).prefetch(2)

val_ds = tf.data.Dataset.from_generator(XRDDataGenerator(FILES["validation"], SERIES_LEN, DEPTH, False), 
                                        output_signature=output_sig).batch(BATCH_SIZE).prefetch(2)

model = build_srdtrans_swin(embed_dim=EMBED_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mae', metrics=['mse'])

print("Training startet...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)