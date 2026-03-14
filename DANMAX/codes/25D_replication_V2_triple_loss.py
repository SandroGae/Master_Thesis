#!/usr/bin/env python3

import os
import sys
import random
import gc
import shutil
import argparse
import json
import time
import socket
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

# TF-Import nach ENV ist sauberer
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# 1. SETUP & ARGUMENT PARSING (20 Jobs Array)
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True, help="Index 0-19 für die 20 Jobs")
args = parser.parse_args()

job_configs = []
# Punkt 02: alpha = 0.0, beta = 2/6
for s in range(42, 52):
    job_configs.append({"point": "P02", "alpha": 0.0, "beta": 2.0/6.0, "seed": s})
# Punkt 14: alpha = 2/6, beta = 0.0
for s in range(42, 52):
    job_configs.append({"point": "P14", "alpha": 2.0/6.0, "beta": 0.0, "seed": s})

if args.task_id < 0 or args.task_id > 19:
    print(f"❌ Ungültige task_id {args.task_id}. Erlaubt sind 0 bis 19.")
    sys.exit(1)

current_config = job_configs[args.task_id]
MY_POINT = current_config["point"]
MY_ALPHA = current_config["alpha"]
MY_BETA = current_config["beta"]
MY_SEED = current_config["seed"]

# REPRODUZIERBARKEIT
os.environ['PYTHONHASHSEED'] = str(MY_SEED)
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.random.set_seed(MY_SEED)
tf.config.experimental.enable_op_determinism()

# PFADE
SCRATCH_ROOT = Path.home() / "scratch" / "DANMAX"
BAMBOO_RAW = Path("/scratch/sgaell/DATA_DANMAX/2026020508/raw/bamboo/")
TB_ROOT = SCRATCH_ROOT / "codes" / "tb_root"

MODEL_OUT_DIR = SCRATCH_ROOT / "models" / f"25D_replication_V2_{MY_POINT}"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
TB_ROOT.mkdir(parents=True, exist_ok=True)

# PARAMETER
DEPTH = 5
SERIES_LEN = 40
BASEFILTERS = 64
CROP_SIZE = (512, 512)
EPOCHS = 200
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 25
RLROP_PATIENCE = 15
BATCH_SIZE = 8
DATA_SPLIT_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# 2. LOCK / STATE HELPERS (Für Array Jobs)
# =====================================================
LOCK_STALE_SECONDS = 2 * 60 * 60 

def write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)

def acquire_lock(lock_file: Path, stale_seconds: int = LOCK_STALE_SECONDS) -> bool:
    now = time.time()
    if lock_file.exists():
        try:
            age = now - lock_file.stat().st_mtime
            if age > stale_seconds: lock_file.unlink(missing_ok=True)
        except: pass
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "created_at_unix": now}, f)
        return True
    except FileExistsError:
        return False

def touch_lock(lock_file: Path):
    try:
        if lock_file.exists(): os.utime(lock_file, None)
    except: pass

def read_state_file(state_file: Path):
    if not state_file.exists(): return None
    try:
        with open(state_file, "r") as f: return json.load(f)
    except: return None

# =====================================================
# 3. METRIKEN & TRIPLE LOSS
# =====================================================
def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_m = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_m)

def display_loss_1000(y_true, y_pred):
    return mae_ssim_2d(y_true, y_pred) * 1000.0

def mae_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

def psnr_clipped(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_clipped(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# =====================================================
# 4. ARCHITEKTUR & DATA UTILS (Aus Replication V2)
# =====================================================
def conv_block_2d(x, filters):
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(512, 512, DEPTH)):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, BASEFILTERS)           ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, BASEFILTERS * 2)           ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, BASEFILTERS * 4)           ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, BASEFILTERS * 8)           ; p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, BASEFILTERS * 16)
    u4 = layers.Conv2DTranspose(BASEFILTERS * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])               ; c5 = conv_block_2d(u4, BASEFILTERS * 8)
    u3 = layers.Conv2DTranspose(BASEFILTERS * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])               ; c6 = conv_block_2d(u3, BASEFILTERS * 4)
    u2 = layers.Conv2DTranspose(BASEFILTERS * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])               ; c7 = conv_block_2d(u2, BASEFILTERS * 2)
    u1 = layers.Conv2DTranspose(BASEFILTERS, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])               ; c8 = conv_block_2d(u1, BASEFILTERS)
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(c8) 
    return models.Model(inputs, out, name="unet_25d_stacked")

def load_and_correct_danmax(base_path, scan_id, crop_size=CROP_SIZE):
    ct_file    = os.path.join(base_path, f"scan-{scan_id:04d}_orca.h5")
    white_file = os.path.join(base_path, f"scan-{scan_id-1:04d}_orca.h5")
    dark_file  = os.path.join(base_path, f"scan-{scan_id-2:04d}_orca.h5")
    data_path  = 'entry/instrument/orca/data'
    with h5py.File(dark_file, 'r') as f_d, h5py.File(white_file, 'r') as f_w, h5py.File(ct_file, 'r') as f_ct:
        m_dark  = np.mean(f_d[data_path][:], axis=0).astype(np.float32)
        m_white = np.mean(f_w[data_path][:], axis=0).astype(np.float32)
        h_s, w_s = (f_ct[data_path].shape[1] - crop_size[0]) // 2, (f_ct[data_path].shape[2] - crop_size[1]) // 2
        projs = f_ct[data_path][:2000, h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]].astype(np.float32)
        m_dark_c  = m_dark[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        m_white_c = m_white[h_s:h_s+crop_size[0], w_s:w_s+crop_size[1]]
        denom = m_white_c - m_dark_c
        denom[denom < 1e-6] = 1e-6
        corrected = (projs - m_dark_c) / denom
        return np.clip(corrected, 0, 1)

def load_split_danmax(base_path, gt_id, low_id):
    return load_and_correct_danmax(base_path, low_id)[..., np.newaxis], load_and_correct_danmax(base_path, gt_id)[..., np.newaxis]

def make_sliding_windows(X, y, series_len, depth):
    N = X.shape[0]; n_series = N // series_len; n_vols = series_len - depth + 1
    X_vols, y_vols = [], []
    for i in range(n_series):
        bx, by = X[i * series_len:(i+1) * series_len], y[i * series_len:(i+1) * series_len]
        for s_idx in range(n_vols):
            X_vols.append(bx[s_idx : s_idx + depth])
            y_vols.append(by[s_idx : s_idx + depth])
    return np.stack(X_vols, axis=0), np.stack(y_vols, axis=0)

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def augment_and_normalize_3d_per_slice(p: float, phys_max: float):
    def map_volume(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x, y = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x), tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        x = x / (tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12)
        y = y / (tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12)
        x = tf.clip_by_value(x / phys_max, 0.0, 1.0)
        y = tf.clip_by_value(y / phys_max, 0.0, 1.0)
        return x, y
    return map_volume

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS: return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

# =====================================================
# 5. MAIN LOOP
# =====================================================
def main():
    RUN_NAME = f"{MY_POINT}__a{MY_ALPHA:.4f}_b{MY_BETA:.4f}_seed{MY_SEED}"
    print(f"\n{'='*60}")
    print(f"🚀 STARTE JOB {args.task_id}/19 | Punkt: {MY_POINT} | Seed: {MY_SEED}")
    print(f"{'='*60}\n")

    lock_file = MODEL_OUT_DIR / f"{RUN_NAME}.lock"
    state_file = MODEL_OUT_DIR / f"{RUN_NAME}__state.json"
    csv_file = MODEL_OUT_DIR / f"{RUN_NAME}_metrics.csv"
    best_keras_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_model.keras"
    best_h5_file = MODEL_OUT_DIR / f"{RUN_NAME}_best_weights.h5"
    temp_checkpoint_file = MODEL_OUT_DIR / f"{RUN_NAME}_TEMP_weights.h5"
    temp_csv = f"temp_{RUN_NAME}.csv"

    state = read_state_file(state_file)
    if state is not None and state.get("status") == "finished":
        print(f"✅ Job {RUN_NAME} bereits erfolgreich abgeschlossen. Beende.")
        sys.exit(0)

    if not acquire_lock(lock_file):
        print(f"⚠️ Lock aktiv. Anderer Prozess trainiert {RUN_NAME} bereits.")
        sys.exit(0)

    try:
        write_json_atomic(state_file, {
            "status": "running", "run_name": RUN_NAME, "point": MY_POINT, "seed": MY_SEED,
            "started_at": datetime.now().isoformat(timespec="seconds")
        })

        print("Lade Daten von DanMAX...")
        X_all, y_all = load_split_danmax(BAMBOO_RAW, 32, 57)
        N_SERIES = len(X_all) // SERIES_LEN

        X_series = np.reshape(X_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_series = np.reshape(y_all, (N_SERIES, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))

        rng = np.random.default_rng(DATA_SPLIT_SEED)
        indices = np.arange(N_SERIES); rng.shuffle(indices)
        X_series, y_series = X_series[indices], y_series[indices]

        n_train, n_val = int(0.6 * N_SERIES), int(0.2 * N_SERIES)

        X_train_raw = np.reshape(X_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_train_raw = np.reshape(y_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        X_val_raw = np.reshape(X_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_val_raw = np.reshape(y_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        
        # Test-Set aufbereiten wie in V2 Referenz
        X_test_raw = np.reshape(X_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_test_raw = np.reshape(y_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

        X_train, y_train = make_sliding_windows(X_train_raw, y_train_raw, SERIES_LEN, DEPTH)
        X_val,   y_val   = make_sliding_windows(X_val_raw,   y_val_raw,   SERIES_LEN, DEPTH)
        X_test,  y_test  = make_sliding_windows(X_test_raw,  y_test_raw,  SERIES_LEN, DEPTH)

        X_train, y_train = shuffle_initial(X_train, y_train, MY_SEED)
        X_val,   y_val   = shuffle_initial(X_val,   y_val,   MY_SEED)
        X_test,  y_test  = shuffle_initial(X_test,  y_test,  MY_SEED)

        print("\nBerechne optimalen Skalierungsfaktor...")
        def get_peak(data):
            sums = np.sum(data, axis=(2, 3, 4), keepdims=True) + 1e-12
            return np.percentile(data / sums, 99.99)
        PHYSICAL_MAX = float(max(get_peak(y_train), get_peak(X_train)) * 1.02)

        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    .shuffle(len(X_train), seed=MY_SEED)
                    .map(augment_and_normalize_3d_per_slice(0.5, PHYSICAL_MAX), num_parallel_calls=AUTOTUNE)
                    .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))

        val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                  .map(augment_and_normalize_3d_per_slice(0.0, PHYSICAL_MAX), num_parallel_calls=AUTOTUNE)
                  .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))
                  
        test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
                   .map(augment_and_normalize_3d_per_slice(0.0, PHYSICAL_MAX), num_parallel_calls=AUTOTUNE)
                   .map(prepare_25d_input, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

        model = unet_2d_stacked()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)
        model.compile(optimizer=optimizer, loss=get_triple_loss(MY_ALPHA, MY_BETA),
                      metrics=[display_loss_1000, mae_clipped, mse_clipped, psnr_clipped, ssim_clipped])

        status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
        def check_crash(epoch, logs):
            psnr = logs.get("val_psnr_clipped", 0)
            if psnr > status["best_psnr"]:
                status["best_psnr"] = psnr
                status["drop_cnt"] = 0
            elif epoch >= 10:
                if psnr < (status["best_psnr"] - 4.5) or psnr < 24.0: status["drop_cnt"] += 1
                if status["drop_cnt"] >= 3:
                    status["aborted"], status["reason"], model.stop_training = True, "perf_collapse", True
            touch_lock(lock_file)

        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE, verbose=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=str(temp_checkpoint_file), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
            tf.keras.callbacks.CSVLogger(temp_csv),
            *tb_callbacks(TB_ROOT / RUN_NAME),
        ]

        print("Training beginnt...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

        term_reason = "psnr_safety_net" if status["aborted"] else ("early_stopping" if len(history.history["loss"]) < EPOCHS else "max_epochs_200")

        print("Evaluation auf dem Test-Set...")
        test_results = model.evaluate(test_ds, verbose=1, return_dict=True)

    except Exception as e:
        status["aborted"], status["reason"], term_reason = True, f"crash_{str(e)[:25]}", "crash"
        test_results = {}

    # --- FINALISIERUNG ---
    if temp_checkpoint_file.exists():
        model.load_weights(str(temp_checkpoint_file))
        model.save(str(best_keras_file))
        model.save_weights(str(best_h5_file))

    best_idx = int(np.argmin(history.history["val_loss"])) if history else 0
    
    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (CROP_SIZE[0], CROP_SIZE[1], DEPTH),
                          extra={"point": MY_POINT, "alpha": MY_ALPHA, "beta": MY_BETA, "seed": MY_SEED,
                                 "aborted": status["aborted"], "reason": status["reason"], "term_reason": term_reason,
                                 "best_epoch": best_idx + 1,
                                 "test_loss": float(test_results.get("loss", -1)),
                                 "test_psnr": float(test_results.get("psnr_clipped", -1))})
    
    if history: finalize_run(model, history, RUN_NAME, meta, folder_name=str(MODEL_OUT_DIR))
    
    if os.path.exists(temp_csv): shutil.move(temp_csv, csv_file)

    write_json_atomic(state_file, {"status": "finished" if term_reason in ["max_epochs_200", "early_stopping", "psnr_safety_net"] else "incomplete",
                                   "run_name": RUN_NAME, "term_reason": term_reason})

    if temp_checkpoint_file.exists(): temp_checkpoint_file.unlink()
    if lock_file.exists(): lock_file.unlink()
    tf.keras.backend.clear_session(); gc.collect()

if __name__ == "__main__":
    main()