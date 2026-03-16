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
# 1. SETUP & ARGUMENT PARSING (8 Jobs: 2 Punkte x 4 Materialien)
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True, help="Index 0-7 für die 8 Jobs")
args = parser.parse_args()

MATERIALS = ["bamboo", "carbon_fiber", "glass_fiber", "chicken_liver"]

job_configs = []
# Punkt 02: alpha = 0.0, beta = 2/6 (Task 0 bis 3)
for mat in MATERIALS:
    job_configs.append({"point": "P02", "alpha": 0.0, "beta": 2.0/6.0, "material": mat})

# Punkt 14: alpha = 2/6, beta = 0.0 (Task 4 bis 7)
for mat in MATERIALS:
    job_configs.append({"point": "P14", "alpha": 2.0/6.0, "beta": 0.0, "material": mat})

if args.task_id < 0 or args.task_id > 7:
    print(f"❌ Ungültige task_id {args.task_id}. Erlaubt sind 0 bis 7.")
    sys.exit(1)

current_config = job_configs[args.task_id]
CURRENT_MATERIAL = current_config["material"]
MY_POINT = current_config["point"]
MY_ALPHA = current_config["alpha"]
MY_BETA = current_config["beta"]
MY_SEED = 42

# REPRODUZIERBARKEIT
os.environ['PYTHONHASHSEED'] = str(MY_SEED)
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.random.set_seed(MY_SEED)
tf.config.experimental.enable_op_determinism()

# PFADE
SCRATCH_ROOT = Path.home() / "scratch" / "DANMAX"
RAW_BASE_DIR = Path("/scratch/sgaell/DATA_DANMAX/2026020508/raw")

# OUTPUT ORDNER FÜR SPEZIALISTEN
TEST_OUT_ROOT = SCRATCH_ROOT / "models" / "TEST_V2_Specialists_512"
# Der Ordner heißt jetzt z.B. "Spezialist_P02_bamboo"
MODEL_OUT_DIR = TEST_OUT_ROOT / f"Spezialist_{MY_POINT}_{CURRENT_MATERIAL}"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

TB_ROOT = TEST_OUT_ROOT / "tb_root"
TB_ROOT.mkdir(parents=True, exist_ok=True)

# PARAMETER
DEPTH = 5
SERIES_LEN = 40
BASEFILTERS = 64
CROP_SIZE = (512, 512) # 512er Crop!
EPOCHS = 200
LR_TARGET = 2e-4
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 25
RLROP_PATIENCE = 15
BATCH_SIZE = 8 # Reduziert für 512x512, um GPU OOM zu verhindern!
DATA_SPLIT_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# MULTI-DATASET CONFIG 
DATASETS = {
    "bamboo":        {"gt_id": 32,  "lc_id": 57},  
    "carbon_fiber":  {"gt_id": 60,  "lc_id": 84},  
    "glass_fiber":   {"gt_id": 87,  "lc_id": 111}, 
    "chicken_liver": {"gt_id": 114, "lc_id": 138}  
}

# =====================================================
# 2. LOCK / STATE HELPERS 
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
            if (now - lock_file.stat().st_mtime) > stale_seconds: lock_file.unlink(missing_ok=True)
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

def mae_clipped(y_true, y_pred): return tf.reduce_mean(tf.abs(tf.clip_by_value(y_true, 0.0, 1.0) - tf.clip_by_value(y_pred, 0.0, 1.0)))
def mse_clipped(y_true, y_pred): return tf.reduce_mean(tf.math.squared_difference(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)))
def psnr_clipped(y_true, y_pred): 
    mse = tf.reduce_mean(tf.math.squared_difference(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0)), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)
def ssim_clipped(y_true, y_pred): return tf.reduce_mean(tf.image.ssim(tf.clip_by_value(y_true, 0.0, 1.0), tf.clip_by_value(y_pred, 0.0, 1.0), max_val=1.0))

# =====================================================
# 4. ARCHITEKTUR & DATA UTILS 
# =====================================================
def conv_block_2d(x, filters):
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape=(CROP_SIZE[0], CROP_SIZE[1], DEPTH)):
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

def get_valid_indices(num_slices, series_len, depth):
    """Gibt nur die Start-Indizes für gültige 5er-Fenster zurück (ohne Serien-Grenzen zu überschreiten)."""
    indices = []
    n_series = num_slices // series_len
    for i in range(n_series):
        start_idx = i * series_len
        # Bei series_len=40 und depth=5 gibt es 36 gültige Fenster pro Serie
        n_vols = series_len - depth + 1
        for s_idx in range(n_vols):
            indices.append(start_idx + s_idx)
    return np.array(indices)


# --- VERBESSERTER ON-THE-FLY GENERATOR ---
def get_dynamic_generator(X_raw, y_raw, indices, depth=DEPTH, shuffle_every_epoch=False):
    def gen():
        # Wir machen eine lokale Kopie der Indizes für diesen Durchlauf
        current_indices = indices.copy()
        if shuffle_every_epoch:
            np.random.shuffle(current_indices)
        
        for idx in current_indices:
            yield X_raw[idx : idx + depth], y_raw[idx : idx + depth]
    return gen

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X)); rng.shuffle(indices)
    return X[indices], y[indices]

def process_data(training=True):
    def map_fn(x, y):
        if training:
            flip = tf.random.uniform([], 0, 1) < 0.5
            x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
            y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)

        x = tf.nn.relu(x)
        y = tf.nn.relu(y)
        
        max_x = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        x = x / max_x
        
        max_y = tf.reduce_max(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        y = y / max_y

        if training:
            scale_factor = tf.random.uniform([], 0.3333, 1.0)
            x = x * scale_factor
            y = y * scale_factor

        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)

        x = tf.transpose(tf.squeeze(x, axis=-1), [1, 2, 0])
        y_center = y[tf.shape(y)[0] // 2]
        
        return x, y_center
    return map_fn

def lr_warmup_scheduler(epoch, lr): 
    return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else lr

# =====================================================
# 5. MAIN LOOP
# =====================================================
def main():
    # Der Name des Modells beinhaltet jetzt klar das Material
    RUN_NAME = f"{MY_POINT}_{CURRENT_MATERIAL}_seed{MY_SEED}_512px"
    print(f"\n{'='*60}")
    print(f"🚀 STARTE SPEZIALISTEN-JOB {args.task_id}/3 | Material: {CURRENT_MATERIAL}")
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
        write_json_atomic(state_file, {"status": "running", "run_name": RUN_NAME, "material": CURRENT_MATERIAL, "point": MY_POINT, "seed": MY_SEED, "started_at": datetime.now().isoformat(timespec="seconds")})

        print(f"Lade Daten für Spezialist: {CURRENT_MATERIAL}...")
        ids = DATASETS[CURRENT_MATERIAL]
        base_p = RAW_BASE_DIR / CURRENT_MATERIAL
        x_data, y_data = load_split_danmax(base_p, ids["gt_id"], ids["lc_id"])
        
        # In Serien umwandeln (Hier wird viel RAM gespart, da wir nur 1 Material laden!)
        n_ser = len(x_data) // SERIES_LEN
        X_series = np.reshape(x_data[:n_ser*SERIES_LEN], (n_ser, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_series = np.reshape(y_data[:n_ser*SERIES_LEN], (n_ser, SERIES_LEN, CROP_SIZE[0], CROP_SIZE[1], 1))
        
        N_TOTAL_SERIES = len(X_series)
        print(f"✅ Insgesamt {N_TOTAL_SERIES} Serien (á {SERIES_LEN} Slices) geladen.")

        rng = np.random.default_rng(DATA_SPLIT_SEED)
        indices = np.arange(N_TOTAL_SERIES); rng.shuffle(indices)
        X_series, y_series = X_series[indices], y_series[indices]

        n_train, n_val = int(0.6 * N_TOTAL_SERIES), int(0.2 * N_TOTAL_SERIES)

        X_train_raw = np.reshape(X_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_train_raw = np.reshape(y_series[:n_train], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        X_val_raw = np.reshape(X_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_val_raw = np.reshape(y_series[n_train:n_train+n_val], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        X_test_raw = np.reshape(X_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))
        y_test_raw = np.reshape(y_series[n_train+n_val:], (-1, CROP_SIZE[0], CROP_SIZE[1], 1))

        # NEUER CODE: Wir berechnen nur noch die "Fahrpläne" (Indizes)
        print("Berechne Start-Indizes für den Generator (spart 80% RAM!)...")
        train_indices = get_valid_indices(len(X_train_raw), SERIES_LEN, DEPTH)
        val_indices = get_valid_indices(len(X_val_raw), SERIES_LEN, DEPTH)
        test_indices = get_valid_indices(len(X_test_raw), SERIES_LEN, DEPTH)

        # Mische nur die Indizes, nicht die riesigen Arrays
        rng = np.random.default_rng(MY_SEED)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        output_sig = (
            tf.TensorSpec(shape=(DEPTH, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(DEPTH, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=tf.float32)
        )

        # Datasets bauen (Geben die Raw-Daten und die Indizes an den Generator)
        train_ds = (tf.data.Dataset.from_generator(
                # HIER NEU: shuffle_every_epoch=True
                get_dynamic_generator(X_train_raw, y_train_raw, train_indices, shuffle_every_epoch=True), 
                output_signature=output_sig)
            .map(process_data(training=True), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
        

        val_ds = (tf.data.Dataset.from_generator(
                get_dynamic_generator(X_val_raw, y_val_raw, val_indices), 
                output_signature=output_sig)
            .map(process_data(training=False), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
                  
        test_ds = (tf.data.Dataset.from_generator(
                get_dynamic_generator(X_test_raw, y_test_raw, test_indices), 
                output_signature=output_sig)
            .map(process_data(training=False), num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
        
        model = unet_2d_stacked()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True, clipnorm=1.0)
        
        model.compile(optimizer=optimizer, loss=get_triple_loss(MY_ALPHA, MY_BETA),
                      metrics=[mae_clipped, mse_clipped, psnr_clipped, ssim_clipped])

        status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
        def check_crash(epoch, logs):
            psnr = logs.get("val_psnr_clipped", 0)
            if psnr > status["best_psnr"]:
                status["best_psnr"] = psnr; status["drop_cnt"] = 0
            elif epoch >= 10:
                if psnr < (status["best_psnr"] - 4.5) or psnr < 15.0: status["drop_cnt"] += 1
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
        term_reason = "psnr_safety_net" if status["aborted"] else ("early_stopping" if len(history.history["loss"]) < EPOCHS else f"max_epochs_{EPOCHS}")

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
                          extra={"material": CURRENT_MATERIAL, "point": MY_POINT, "alpha": MY_ALPHA, "beta": MY_BETA, "seed": MY_SEED,
                                 "aborted": status["aborted"], "reason": status["reason"], "term_reason": term_reason,
                                 "best_epoch": best_idx + 1, "test_loss": float(test_results.get("loss", -1)),
                                 "test_psnr": float(test_results.get("psnr_clipped", -1))})
    
    if history: finalize_run(model, history, RUN_NAME, meta, folder_name=str(MODEL_OUT_DIR))
    if os.path.exists(temp_csv): shutil.move(temp_csv, csv_file)

    write_json_atomic(state_file, {"status": "finished" if term_reason in [f"max_epochs_{EPOCHS}", "early_stopping", "psnr_safety_net"] else "incomplete",
                                   "run_name": RUN_NAME, "term_reason": term_reason})

    if temp_checkpoint_file.exists(): temp_checkpoint_file.unlink()
    if lock_file.exists(): lock_file.unlink()
    tf.keras.backend.clear_session(); gc.collect()

if __name__ == "__main__":
    main()