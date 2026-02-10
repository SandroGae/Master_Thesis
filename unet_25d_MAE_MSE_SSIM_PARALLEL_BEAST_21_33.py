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
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# GPU Optimierung
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# =====================================================
# 1. SETUP & KONFIGURATION
# =====================================================
if len(sys.argv) < 2:
    print("Fehler: Punkt-Index (0 oder 1) muss übergeben werden!")
    sys.exit(1)

POINT_IDX = int(sys.argv[1])
TARGET_POINTS = [(0.0, 0.0), (round(10/12, 4), 0.0)]
MY_ALPHA, MY_BETA = TARGET_POINTS[POINT_IDX]

SUCCESS_GOAL = 9  
START_SEED = 43 

DEPTH = 5
BATCH_SIZE = 8
EPOCHS = 200

ROOT_DATA = Path.home() / "data"
SEED_STUDY_ROOT = ROOT_DATA / "seed_study_infinite" 
SUCCESS_DIR = SEED_STUDY_ROOT / f"success_point_{POINT_IDX}"
FAILED_DIR = SEED_STUDY_ROOT / f"failed_point_{POINT_IDX}"
TB_ROOT = SEED_STUDY_ROOT / "tblogs"

for d in [SUCCESS_DIR, FAILED_DIR, TB_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. METRIKEN & LOSS (Erweitert)
# =====================================================
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

def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        yt = tf.cast(yt, tf.float32); yp = tf.cast(yp, tf.float32)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

# --- Architektur (Identisch) ---
def conv_block_2d(x, filters):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=ki)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, 5)):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block_2d(inputs, 64); p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, 128);    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, 256);    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, 512);    p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, 1024)
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); c5 = conv_block_2d(u4, 512)
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]); c6 = conv_block_2d(u3, 256)
    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]); c7 = conv_block_2d(u2, 128)
    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]); c8 = conv_block_2d(u1, 64)
    out = layers.Conv2D(1, (1, 1), activation="sigmoid")(c8)
    return models.Model(inputs, out)

# =====================================================
# 3. DATA UTILITIES (Identisch)
# =====================================================
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        lc = f["low_count/data"][:].astype('float32')
        hc = f["high_count/data"][:].astype('float32')
    lc = np.moveaxis(lc, -1, 0)[:, :, :, np.newaxis]
    hc = np.moveaxis(hc, -1, 0)[:, :, :, np.newaxis]
    return lc, hc

def make_sliding_windows(X, y, series_len, depth):
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(X.shape[0] // series_len):
        bx = X[i*series_len : (i+1)*series_len]
        by = y[i*series_len : (i+1)*series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s+depth]); y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
        y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sx = tf.reduce_sum(tf.nn.relu(x), [1,2,3], keepdims=True) + 1e-12
        sy = tf.reduce_sum(tf.nn.relu(y), [1,2,3], keepdims=True) + 1e-12
        sc = tf.random.uniform([], scale_min, scale_max)
        return (x/sx)*sc, (y/sy)*sc
    return map_vol

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

# =====================================================
# 4. INFINITE LOOP
# =====================================================
print(f"Lade Daten für Punkt Alpha={MY_ALPHA}, Beta={MY_BETA}...")
FILES = {"training": "/home/sgaell/data/original_data/training_data.hdf5", 
         "validation": "/home/sgaell/data/original_data/validation_data.hdf5"}
X_train_raw, y_train_raw = load_split(FILES["training"])
X_val_raw, y_val_raw = load_split(FILES["validation"])
X_train_win, y_train_win = make_sliding_windows(X_train_raw, y_train_raw, 41, 5)
X_val_win, y_val_win = make_sliding_windows(X_val_raw, y_val_raw, 41, 5)

success_count = 0
current_seed = START_SEED

while success_count < SUCCESS_GOAL:
    existing = list(SUCCESS_DIR.glob(f"*_seed{current_seed}_*"))
    if existing:
        print(f"Seed {current_seed} bereits vorhanden. Überspringe..."); success_count += 1; current_seed += 1; continue

    os.environ['PYTHONHASHSEED'] = str(current_seed)
    random.seed(current_seed); np.random.seed(current_seed); tf.random.set_seed(current_seed)
    tf.config.experimental.enable_op_determinism()
    
    TS_RUN = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_NAME = f"InfSeed_P{POINT_IDX}_a{MY_ALPHA:.4f}_b{MY_BETA:.4f}_seed{current_seed}_{TS_RUN}"
    print(f"\n>>> VERSUCH {success_count + 1}/{SUCCESS_GOAL} | Seed: {current_seed}")

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train_win, y_train_win))
                .shuffle(len(X_train_win), seed=current_seed)
                .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), -1)
                .map(prepare_25d_input, -1).batch(BATCH_SIZE).prefetch(-1))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val_win, y_val_win))
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
                status["aborted"] = True; model.stop_training = True

    temp_csv = f"temp_{RUN_NAME}.csv"
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
        tf.keras.callbacks.CSVLogger(temp_csv),
        *tb_callbacks(TB_ROOT / RUN_NAME)
    ]

    # HIER SIND ALLE METRIKEN FÜR DIE CSV
    model.compile(optimizer=optimizer, loss=get_triple_loss(MY_ALPHA, MY_BETA), 
                  metrics=[mae_center, mse_center, ssim_center, psnr_center])
    
    try:
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
        val_psnr = history.history['val_psnr_center'][-1]
    except Exception as e:
        print(f"Crash: {e}"); val_psnr = 0; status["aborted"] = True

    is_failed = status["aborted"] or val_psnr < 25.0
    target_dir = FAILED_DIR if is_failed else SUCCESS_DIR
    
    if not is_failed: success_count += 1
    
    # METADATEN UPDATE
    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), 
                          extra={
                              "alpha": MY_ALPHA, 
                              "beta": MY_BETA, 
                              "seed": current_seed, 
                              "aborted": status["aborted"],
                              "final_val_psnr": float(val_psnr),
                              "final_val_mae": float(history.history['val_mae_center'][-1]) if not status['aborted'] else 0
                          })
    
    finalize_run(model, history, RUN_NAME, meta, folder_name=str(target_dir))
    if os.path.exists(temp_csv): shutil.move(temp_csv, target_dir / f"{RUN_NAME}.csv")
    
    current_seed += 1
    tf.keras.backend.clear_session(); gc.collect()

print(f"\n--- MISSION COMPLETE: 9 Erfolge für Punkt {POINT_IDX} ---")