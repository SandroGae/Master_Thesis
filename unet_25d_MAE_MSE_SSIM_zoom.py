#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module (müssen im selben Ordner liegen)
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
BATCH_SIZE = 8
EPOCHS = 100
CKPT_FOLDER = "checkpoints_unet"

# DER DEEP-SCAN FOKUS
ALPHA_FIX = 1.5 / 7  # ≈ 0.21428 (Exakt zwischen 1/7 und 2/7)
BETA_LIST = np.linspace(0.0, 1.0, 14) # 14 gleichmäßige Schritte

# --- LEARNING RATE WARMUP (10 Epochen) ---
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 10
    base_lr = 5e-4
    if epoch < warmup_epochs:
        # Linearer Anstieg von 1/10 bis 10/10 der base_lr
        return base_lr * (epoch + 1) / warmup_epochs
    return lr

# --- ARCHITEKTUR ---
def conv_block_2d(x, filters):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH)):
    inputs = layers.Input(shape=input_shape, name="input")
    # Encoder
    c1 = conv_block_2d(inputs, 64)   ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, 128)      ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, 256)      ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, 512)      ; p4 = layers.MaxPooling2D((2, 2))(c4)
    # Bottleneck
    bn = conv_block_2d(p4, 1024)
    # Decoder
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); c5 = conv_block_2d(u4, 512)
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]); c6 = conv_block_2d(u3, 256)
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]); c7 = conv_block_2d(u2, 128)
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]); c8 = conv_block_2d(u1, 64)
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(c8)
    return models.Model(inputs, out, name="unet_25d_stacked")

# --- DATA LOADING & PREP ---
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

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, -1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# --- LOSS & METRICS (CLIPPED!) ---
def get_triple_loss(a, b):
    def loss(yt, yp):
        yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim_l = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        # Kaskadierte Gewichtung
        w_ssim = a
        w_mse  = (1.0 - a) * b
        w_mae  = (1.0 - a) * (1.0 - b)
        return (w_ssim * ssim_l) + (w_mse * mse) + (w_mae * mae)
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

# --- DATA PREPARATION ---
print("Lade Daten...")
X_train, y_train = load_split("/home/sgaell/data/original_data/training_data.hdf5")
X_val, y_val = load_split("/home/sgaell/data/original_data/validation_data.hdf5")
X_train, y_train = make_sliding_windows(X_train, y_train, 41, 5)
X_val, y_val = make_sliding_windows(X_val, y_val, 41, 5)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train.astype('float32'), y_train.astype('float32')))
            .shuffle(len(X_train), seed=SEED).map(prepare_25d_input).batch(BATCH_SIZE).prefetch(-1))
val_ds = (tf.data.Dataset.from_tensor_slices((X_val.astype('float32'), y_val.astype('float32')))
          .map(prepare_25d_input).cache().batch(BATCH_SIZE).prefetch(-1))

# --- TRAINING LOOP ---
for b_idx, beta_val in enumerate(BETA_LIST):
    a_r, b_r = round(float(ALPHA_FIX), 4), round(float(beta_val), 4)
    
    # Eindeutiger Name für diesen Run im Deep-Scan
    RUN_NAME = f"unet_25d_DeepScan_a{a_r}_b{b_r}_bf64_D5"
    TB_DIR = Path.home() / "data" / "tblogs_unet_3d_simple" / RUN_NAME
    TB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"DEEP-SCAN BETA: {b_idx+1}/14 | Alpha={a_r} | Beta={b_r}")
    print("="*60)

    model = unet_2d_stacked()
    # Wichtig: clipnorm für Stabilität
    optimizer = tf.keras.optimizers.Adam(5e-4, amsgrad=True, clipnorm=1.0)
    
    def check_crash(epoch, logs):
        if logs.get('val_psnr_center', 30) < 20.0: 
            print(f"\n[CRASH] PSNR < 20.0 in Epoche {epoch}. Breche ab.")
            model.stop_training = True

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler), # Hier ist dein Warmup!
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
        tf.keras.callbacks.CSVLogger(str(TB_DIR / f"log_{RUN_NAME}.csv")),
        make_epoch_ckpt_callback(RUN_NAME, folder_name=CKPT_FOLDER),
        *tb_callbacks(TB_DIR)
    ]

    model.compile(optimizer=optimizer, loss=get_triple_loss(a_r, b_r), 
                  metrics=[mae_center, mse_center, psnr_center, ssim_center])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    
    # Metadaten für die spätere Analyse
    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), 
                          extra={"alpha": a_r, "beta": b_r, "type": "DeepScan_CentralLine"})
    
    finalize_run(model, history, RUN_NAME, meta, folder_name=CKPT_FOLDER)
    
    # Speicher freigeben für den nächsten Run im Loop
    tf.keras.backend.clear_session()
    import gc; gc.collect()

print("\n--- Deep-Scan erfolgreich beendet ---")