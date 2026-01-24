#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# --- REPRODUZIERBARKEIT ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# --- PARAMETER ---
DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64
BATCH_SIZE = 16
EPOCHS = 200 # Maximale Laufzeit
PATIENCE = 20 # Early Stopping nach 20 Epochen ohne Verbesserung
CKPT_FOLDER = "checkpoints_unet"

# DER DEEP-SCAN FOKUS
ALPHA_FIX = 1.5 / 7
BETA_LIST = np.linspace(0.0, 1.0, 14)

# --- LEARNING RATE WARMUP ---
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 3
    base_lr = 3e-4 # Smaller for better convergence
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return lr

# --- ARCHITEKTUR ---
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, base_filters)          ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2)          ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4)          ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8)          ; p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, base_filters * 16)
    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])               ; c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])               ; c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])               ; c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])               ; c8 = conv_block_2d(u1, base_filters)
    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8)
    return models.Model(inputs, out, name="unet_25d_stacked")

# --- DATA UTILS ---
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low, high = f["low_count/data"][:], f["high_count/data"][:]
    low, high = np.moveaxis(low, -1, 0), np.moveaxis(high, -1, 0)
    return low[:, :, :, np.newaxis], high[:, :, :, np.newaxis]

def make_sliding_windows(X, y, series_len, depth):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bx, by = X[start:start+series_len], y[start:start+series_len]
        for s in range(n_vols_per_series):
            X_v.append(bx[s:s+depth]); y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    return X[idx], y[idx]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, [1,2,3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, [1,2,3], keepdims=True) + 1e-12
        x, y = x/sum_x, y/sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x*scale, y*scale
    return map_vol

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, -1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# --- LOSS & METRICS ---
def get_triple_loss(a, b):
    def loss(yt, yp):
        yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim_l = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (a * ssim_l) + ((1.0 - a) * b * mse) + ((1.0 - a) * (1.0 - b) * mae)
    return loss

def mae_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0.0, 1.0), tf.clip_by_value(yp, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(yt - yp))

def mse_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0.0, 1.0), tf.clip_by_value(yp, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(yt, yp))

def psnr_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0.0, 1.0), tf.clip_by_value(yp, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(yt, yp), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0.0, 1.0), tf.clip_by_value(yp, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))

# --- DATA PREP ---
print("Lade Daten...")
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5", "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}
X_train, y_train = load_split(FILES["training"])
X_val, y_val = load_split(FILES["validation"])
X_train, y_train = make_sliding_windows(X_train, y_train, 41, 5)
X_val, y_val = make_sliding_windows(X_val, y_val, 41, 5)
X_train, y_train = shuffle_initial(X_train, y_train, SEED)
X_val, y_val = shuffle_initial(X_val, y_val, SEED)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train.astype('float32'), y_train.astype('float32')))
            .shuffle(len(X_train), seed=SEED)
            .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), num_parallel_calls=-1)
            .map(prepare_25d_input, num_parallel_calls=-1).batch(BATCH_SIZE).prefetch(-1))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val.astype('float32'), y_val.astype('float32')))
          .map(augment_and_normalize_3d_per_slice(10000, 10001, 0), num_parallel_calls=-1)
          .map(prepare_25d_input, num_parallel_calls=-1).cache().batch(BATCH_SIZE).prefetch(-1))

# --- TRAINING LOOP ---
for b_idx, beta_val in enumerate(BETA_LIST):
    a_r, b_r = round(float(ALPHA_FIX), 4), round(float(beta_val), 4)
    RUN_NAME = f"unet_25d_DeepScan_a{a_r}_b{b_r}_bf64_D5"
    TB_DIR = Path.home() / "data" / "tblogs_unet_3d_simple" / RUN_NAME
    TB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRUN {b_idx+1}/14 | Alpha={a_r} | Beta={b_r}\n" + "="*40)

    model = unet_2d_stacked()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, amsgrad=True, clipnorm=1.0)
    
    def check_crash(epoch, logs):
        if logs.get('val_psnr_center', 30) < 20.0: model.stop_training = True

    # Callbacks inkl. Early Stopping
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
        # --- NEU: Early Stopping ---
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            verbose=1, 
            restore_best_weights=True # Setzt das Modell auf den Stand des niedrigsten val_loss zurück
        ),
        tf.keras.callbacks.CSVLogger(str(TB_DIR / f"log_{RUN_NAME}.csv")),
        make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
        *tb_callbacks(TB_DIR)
    ]

    model.compile(optimizer=optimizer, loss=get_triple_loss(a_r, b_r), 
                  metrics=[mae_center, mse_center, psnr_center, ssim_center])

    # Fit startet
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    
    # Da finalize_run hier außerhalb von model.fit steht, wird es 
    # IMMER ausgeführt, egal ob fit regulär fertig wurde oder abgebrochen ist.
    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 3e-4, (192, 240, 5), 
                          extra={"alpha": a_r, "beta": b_r, "type": "DeepScan_CentralLine"})
    finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
    
    tf.keras.backend.clear_session()
    import gc; gc.collect()

print("\n--- Deep-Scan erfolgreich beendet ---")