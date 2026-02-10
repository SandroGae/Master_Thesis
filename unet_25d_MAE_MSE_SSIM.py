#!/usr/bin/env python3
import os
import random
import gc
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module (Stelle sicher, dass diese im PYTHONPATH liegen)
from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# =====================================================
# 1. PARAMETER & PFAD-SETUP
# =====================================================
DEPTH = 5
SERIES_LEN = 41
BASEFILTERS = 64
BATCH_SIZE = 8
EPOCHS = 200

# Ordner für die Ergebnisse
ROOT_DATA = Path.home() / "data"
SUCCESS_DIR = ROOT_DATA / "new_rerun"
FAILED_DIR = ROOT_DATA / "failed_new_rerun"

for d in [SUCCESS_DIR, FAILED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Grid Setup: 0 bis 1 in 1/6 Schritten (7 Punkte)
GRID_VALS = np.linspace(0.0, 1.0, 7)

# =====================================================
# 2. ARCHITEKTUR & LOSS & METRIKEN
# =====================================================
def conv_block_2d(x, filters, kernel_size=(3, 3), padding="same"):
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer="he_normal", use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS, output_activation="sigmoid"):
    inputs = layers.Input(shape=input_shape, name="input")
    # Encoder
    c1 = conv_block_2d(inputs, base_filters); p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2); p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4); p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8); p4 = layers.MaxPooling2D((2, 2))(c4)
    # Bottleneck
    bn = conv_block_2d(p4, base_filters * 16)
    # Decoder
    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); c5 = conv_block_2d(u4, base_filters * 8)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]); c6 = conv_block_2d(u3, base_filters * 4)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]); c7 = conv_block_2d(u2, base_filters * 2)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]); c8 = conv_block_2d(u1, base_filters)

    out = layers.Conv2D(1, (1, 1), activation=output_activation, name="output")(c8)
    return models.Model(inputs, out, name="unet_25d_stacked")

def get_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        # Kaskadierte Gewichtung
        w_ssim = alpha
        w_mse  = (1.0 - alpha) * beta
        w_mae  = (1.0 - alpha) * (1.0 - beta)
        return (w_ssim * ssim_loss) + (w_mse * mse) + (w_mae * mae)
    return loss

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
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

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
        low_count = f["low_count/data"][:]
        high_count = f["high_count/data"][:]
    low_count = np.moveaxis(low_count, -1, 0)[:, :, :, np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[:, :, :, np.newaxis]
    return low_count, high_count

def make_sliding_windows(X, y, series_len, depth):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_volumes, y_volumes = [], []
    for i in range(n_series):
        start = i * series_len
        blockX, blockY = X[start:start+series_len], y[start:start+series_len]
        for s in range(n_vols_per_series):
            X_volumes.append(blockX[s : s + depth])
            y_volumes.append(blockY[s : s + depth])
    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0.0, 1.0) < p
        x = tf.cond(flip, lambda: tf.reverse(x, axis=[2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, axis=[2]), lambda: y)
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        scale = tf.random.uniform([], scale_min, scale_max)
        return (x / sum_x) * scale, (y / sum_y) * scale
    return map_volume

def prepare_25d_input(x, y):
    x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
    y_center = y[tf.shape(y)[0] // 2]
    return x, y_center

# =====================================================
# 4. TRAINING WORKFLOW
# =====================================================
print("Lade Daten...")
H5_TRAIN = "/home/sgaell/data/original_data/training_data.hdf5"
H5_VAL = "/home/sgaell/data/original_data/validation_data.hdf5"

X_train_raw, y_train_raw = load_split(H5_TRAIN)
X_val_raw, y_val_raw = load_split(H5_VAL)

X_train_win, y_train_win = make_sliding_windows(X_train_raw, y_train_raw, SERIES_LEN, DEPTH)
X_val_win, y_val_win = make_sliding_windows(X_val_raw, y_val_raw, SERIES_LEN, DEPTH)

for alpha in GRID_VALS:
    for beta in GRID_VALS:
        # Redundanz-Check: Wenn alpha=1.0, ist beta irrelevant. Nur beta=0.0 ausführen.
        if alpha == 1.0 and beta > 0.0:
            continue

        run_success = False
        # Retry-Loop für Seeds 42 bis 45
        for current_seed in range(42, 46):
            if run_success: break
            
            # Seed setzen für Reproduzierbarkeit
            os.environ['PYTHONHASHSEED'] = str(current_seed)
            random.seed(current_seed)
            np.random.seed(current_seed)
            tf.random.set_seed(current_seed)
            
            RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
            BASE_RUN_NAME = f"DeepScan_a{alpha:.4f}_b{beta:.4f}_seed{current_seed}"
            FULL_RUN_NAME = f"{BASE_RUN_NAME}_{RUN_TIMESTAMP}"

            print(f"\n" + "="*70)
            print(f"STARTE: Alpha={alpha:.4f}, Beta={beta:.4f}, Seed={current_seed}")
            print(f"Run: {FULL_RUN_NAME}")
            print("="*70)

            # Datasets pro Seed neu shuffeln
            X_t, y_t = shuffle_initial(X_train_win, y_train_win, current_seed)
            train_ds = (tf.data.Dataset.from_tensor_slices((X_t.astype(np.float32), y_t.astype(np.float32)))
                        .shuffle(len(X_t), seed=current_seed, reshuffle_each_iteration=True)
                        .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p=0.5), num_parallel_calls=-1)
                        .map(prepare_25d_input, num_parallel_calls=-1)
                        .batch(BATCH_SIZE).prefetch(-1))

            val_ds = (tf.data.Dataset.from_tensor_slices((X_val_win.astype(np.float32), y_val_win.astype(np.float32)))
                      .map(augment_and_normalize_3d_per_slice(10000.0, 10001.0, p=0.0), num_parallel_calls=-1)
                      .map(prepare_25d_input, num_parallel_calls=-1)
                      .cache().batch(BATCH_SIZE).prefetch(-1))

            # Modell & Crash Detection
            model = unet_2d_stacked()
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
            
            status = {"aborted": False, "best_psnr": -1.0}
            def monitor_psnr(epoch, logs):
                curr_psnr = logs.get('val_psnr_center', 0)
                if curr_psnr > status["best_psnr"]: status["best_psnr"] = curr_psnr
                # Abbruchlogik ab Epoche 10
                if epoch >= 10:
                    if curr_psnr < (status["best_psnr"] - 4.0) or curr_psnr < 27.0:
                        print(f"\n[CRASH DETECTED] Epoch {epoch}: PSNR {curr_psnr:.2f} (Best: {status['best_psnr']:.2f})")
                        status["aborted"] = True
                        model.stop_training = True

            # Callbacks
            temp_csv = "temp_training_log.csv"
            current_callbacks = [
                tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=monitor_psnr),
                tf.keras.callbacks.CSVLogger(temp_csv),
                *tb_callbacks(Path.home() / "data" / "tblogs_new_rerun" / FULL_RUN_NAME)
            ]

            model.compile(optimizer=optimizer, loss=get_triple_loss(alpha, beta), 
                          metrics=[mae_center, mse_center, psnr_center, ssim_center])

            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=current_callbacks, verbose=2)

            # Meta-Daten für finalize_run
            meta = make_meta_dict(FULL_RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), 
                                  extra={"alpha": alpha, "beta": beta, "seed": current_seed, "aborted": status["aborted"]})

            # Finalisierung & Datei-Organisation
            if status["aborted"]:
                print(f">>> RUN GESCHEITERT (Seed {current_seed}). Verschiebe in failed_new_rerun.")
                fail_name = f"failed_{FULL_RUN_NAME}"
                finalize_run(model, history, fail_name, meta, folder_name=str(FAILED_DIR))
                if os.path.exists(temp_csv):
                    shutil.move(temp_csv, FAILED_DIR / f"{fail_name}.csv")
            else:
                print(f">>> RUN ERFOLGREICH. Verschiebe in new_rerun.")
                finalize_run(model, history, FULL_RUN_NAME, meta, folder_name=str(SUCCESS_DIR))
                if os.path.exists(temp_csv):
                    shutil.move(temp_csv, SUCCESS_DIR / f"{FULL_RUN_NAME}.csv")
                run_success = True

            # RAM & VRAM Cleanup
            tf.keras.backend.clear_session()
            gc.collect()

print("\n--- Gesamter Grid-Scan abgeschlossen ---")