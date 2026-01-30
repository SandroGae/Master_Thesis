#!/usr/bin/env python3
import os
import sys
import random
import gc
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# =====================================================
# 1. ARGUMENTE & PFAD-SETUP (Parallel-Logik)
# =====================================================
if len(sys.argv) < 3:
    print("Fehler: Start- und End-Index müssen übergeben werden!")
    sys.exit(1)

START_IDX = int(sys.argv[1])
END_IDX = int(sys.argv[2])

DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64
BATCH_SIZE = 8
EPOCHS = 200
PATIENCE = 25

# Ordner-Struktur
ROOT_DATA = Path.home() / "data"
SUCCESS_DIR = ROOT_DATA / "new_rerun"
FAILED_DIR = ROOT_DATA / "failed_new_rerun"
TB_ROOT = ROOT_DATA / "tblogs_new_rerun"

for d in [SUCCESS_DIR, FAILED_DIR, TB_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# Grid Setup: 43 relevante Kombinationen
GRID_VALS = np.linspace(0.0, 1.0, 7)
ALL_COMBOS = []
for a_val in GRID_VALS:
    for b_val in GRID_VALS:
        if round(float(a_val), 4) == 1.0 and round(float(b_val), 4) > 0.0:
            continue
        ALL_COMBOS.append((round(float(a_val), 4), round(float(b_val), 4)))

MY_COMBOS = ALL_COMBOS[START_IDX:END_IDX]

# =====================================================
# 2. ARCHITEKTUR & LOSS & METRIKEN (Reines Float 32)
# =====================================================
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, base_filters); p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2); p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4); p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8); p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, base_filters * 16)
    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]); c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]); c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]); c8 = conv_block_2d(u1, base_filters)
    # Output Schicht (Standard Float 32)
    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8)
    return models.Model(inputs, out, name="unet_25d_stacked")

def get_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return (alpha * ssim_loss) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

def mae_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.abs(yt - yp))

def mse_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.square(yt - yp))

def psnr_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    mse = tf.reduce_mean(tf.square(yt - yp), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(yt, yp):
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))

# =====================================================
# 3. WARM-UP & DATA UTILITIES
# =====================================================
def lr_warmup_scheduler(epoch, lr):
    warmup_epochs = 10
    base_lr = 5e-4 
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return lr

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low, high = f["low_count/data"][:], f["high_count/data"][:]
    low, high = np.moveaxis(low, -1, 0), np.moveaxis(high, -1, 0)
    return low[:, :, :, np.newaxis], high[:, :, :, np.newaxis]

def make_sliding_windows(X, y, series_len, depth):
    n_series = X.shape[0] // series_len
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(n_series):
        start = i * series_len
        bx, by = X[start:start+series_len], y[start:start+series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s+depth]); y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X)); rng.shuffle(idx)
    return X[idx], y[idx]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sum_x = tf.reduce_sum(tf.nn.relu(x), [1,2,3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(tf.nn.relu(y), [1,2,3], keepdims=True) + 1e-12
        scale = tf.random.uniform([], scale_min, scale_max)
        return (x/sum_x)*scale, (y/sum_y)*scale
    return map_vol

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

# =====================================================
# 4. TRAINING WORKFLOW (GRID SWEEP)
# =====================================================
print("Lade Daten...")
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5", 
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}
X_train_raw, y_train_raw = load_split(FILES["training"])
X_val_raw, y_val_raw = load_split(FILES["validation"])
X_train_win, y_train_win = make_sliding_windows(X_train_raw, y_train_raw, 41, 5)
X_val_win, y_val_win = make_sliding_windows(X_val_raw, y_val_raw, 41, 5)

for a_r, b_r in MY_COMBOS:
    current_seed = 42
    run_finished = False
    
    while not run_finished:
        os.environ['PYTHONHASHSEED'] = str(current_seed)
        random.seed(current_seed); np.random.seed(current_seed); tf.random.set_seed(current_seed)
        tf.config.experimental.enable_op_determinism()
        
        TS_RUN = datetime.now().strftime("%Y%m%d-%H%M%S")
        RUN_NAME = f"DeepScan_a{a_r:.4f}_b{b_r:.4f}_seed{current_seed}_{TS_RUN}"
        print(f"\n>>> START: Alpha={a_r}, Beta={b_r}, Seed={current_seed}")
        
        X_t, y_t = shuffle_initial(X_train_win, y_train_win, current_seed)
        train_ds = (tf.data.Dataset.from_tensor_slices((X_t.astype('float32'), y_t.astype('float32')))
                    .shuffle(len(X_t), seed=current_seed)
                    .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), -1)
                    .map(prepare_25d_input, -1).batch(BATCH_SIZE).prefetch(-1))

        val_ds = (tf.data.Dataset.from_tensor_slices((X_val_win.astype('float32'), y_val_win.astype('float32')))
                  .map(augment_and_normalize_3d_per_slice(10000, 10001, 0), -1)
                  .map(prepare_25d_input, -1).cache().batch(BATCH_SIZE).prefetch(-1))

        model = unet_2d_stacked()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
        
        status = {"aborted": False, "best_psnr": -1.0}
        def check_crash(epoch, logs):
            psnr = logs.get('val_psnr_center', 0)
            if psnr > status["best_psnr"]: status["best_psnr"] = psnr
            if epoch >= 10:
                if psnr < (status["best_psnr"] - 4.0) or psnr < 27.0:
                    print(f"\n[CRASH] PSNR {psnr:.2f} (Best: {status['best_psnr']:.2f}). Aborting.")
                    status["aborted"] = True; model.stop_training = True

        temp_csv = f"temp_{RUN_NAME}.csv"
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
            tf.keras.callbacks.CSVLogger(temp_csv),
            *tb_callbacks(TB_ROOT / RUN_NAME)
        ]

        model.compile(optimizer=optimizer, loss=get_triple_loss(a_r, b_r), metrics=[mae_center, mse_center, psnr_center, ssim_center])
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

        # EVALUATION & DATEI-LOGIK
        last_psnr = history.history['val_psnr_center'][-1]
        failed = status["aborted"] or last_psnr < 25.0
        
        if failed:
            final_name = f"failed_{RUN_NAME}"
            meta = make_meta_dict(final_name, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), extra={"alpha": a_r, "beta": b_r, "seed": current_seed, "aborted": True})
            finalize_run(model, history, final_name, meta, folder_name=str(FAILED_DIR))
            if os.path.exists(temp_csv): shutil.move(temp_csv, FAILED_DIR / f"{final_name}.csv")
            
            if current_seed < 45:
                print(f"\n!!! RETRYING WITH SEED {current_seed + 1} !!!")
                current_seed += 1
            else:
                print(f"\n!!! GRID POINT FAILED TWICE. Skipping...")
                run_finished = True
        else:
            meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), extra={"alpha": a_r, "beta": b_r, "seed": current_seed, "aborted": False})
            finalize_run(model, history, RUN_NAME, meta, folder_name=str(SUCCESS_DIR))
            if os.path.exists(temp_csv): shutil.move(temp_csv, SUCCESS_DIR / f"{RUN_NAME}.csv")
            run_finished = True

        tf.keras.backend.clear_session(); gc.collect()

print("\n--- BEAST MODE SCAN FINISHED ---")