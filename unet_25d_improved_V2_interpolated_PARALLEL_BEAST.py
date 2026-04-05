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
# 1. SETUP & ARGUMENT PARSING (SEED PRO JOB)
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Seed für diesen spezifischen Job (z.B. 42)")
args = parser.parse_args()

current_seed = args.seed

# REPRODUZIERBARKEIT (GLOBAL FÜR DIESEN JOB)
os.environ['PYTHONHASHSEED'] = str(current_seed)
random.seed(current_seed)
np.random.seed(current_seed)
tf.random.set_seed(current_seed)
tf.config.experimental.enable_op_determinism() # Wichtig für A100 Determinismus

# HARDCODED POINT 14
MY_POINT_IDX = 14
MY_ALPHA = 2.0 / 6.0
MY_BETA = 0.0

SCRATCH_ROOT = Path.home() / "scratch" / "43_Models_10_Seeds"
TB_ROOT = SCRATCH_ROOT / "tensorboard_logs"
DATA_ROOT = Path.home() / "data"
ORIGINAL_DATA_DIR = DATA_ROOT / "original_data"
INTERP_DIR = DATA_ROOT / "interpolated_data_linear"

for d in [SCRATCH_ROOT, TB_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# GPU Memory Growth robust aktivieren (zusaetzlich zum ENV)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"Warnung: Konnte GPU memory growth nicht setzen: {e}")

AUTOTUNE = tf.data.AUTOTUNE

DEPTH = 5
SERIES_LEN_INTERP = 241
SERIES_LEN_ORIG = 41
BATCH_SIZE = 8
EPOCHS = 200
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
USE_POISSON_NOISE = True

# =====================================================
# LOCK / STATE HELPERS (robust fuer Slurm Array Jobs)
# =====================================================
LOCK_STALE_SECONDS = 2 * 60 * 60  # 2h ohne Heartbeat => stale lock


def write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def cleanup_run_files(point_dir: Path, run_name: str, keep_names=None):
    keep_names = keep_names or set()
    for p in point_dir.glob(f"{run_name}*"):
        try:
            if p.is_file() and p.name not in keep_names:
                p.unlink()
        except Exception as e:
            print(f"Warnung: Konnte {p} nicht loeschen: {e}")


def acquire_lock(lock_file: Path, stale_seconds: int = LOCK_STALE_SECONDS) -> bool:
    now = time.time()
    if lock_file.exists():
        try:
            age = now - lock_file.stat().st_mtime
            if age > stale_seconds:
                print(f"Stale lock entfernt: {lock_file.name} (Alter {age/3600:.1f}h)")
                lock_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warnung: Konnte Lock nicht pruefen/loeschen ({lock_file}): {e}")

    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump({
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "created_at_unix": now,
                "created_at_iso": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
        return True
    except FileExistsError:
        return False


def touch_lock(lock_file: Path):
    try:
        if lock_file.exists():
            os.utime(lock_file, None)
    except Exception:
        pass


def read_state_file(state_file: Path):
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception:
        return None
    

# =====================================================
# 2. METRIKEN & LOSS (MIT CLIPPING)
# =====================================================
def mae_clipped(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_clipped(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))

def psnr_clipped(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_clipped(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss


# =====================================================
# 3. ARCHITEKTUR & DATA UTILS
# =====================================================
def conv_block_2d(x, filters):
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
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


def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        lc = f["low_count/data"][:].astype("float32")
        hc = f["high_count/data"][:].astype("float32")
    lc = np.moveaxis(lc, -1, 0)[:, :, :, np.newaxis]
    hc = np.moveaxis(hc, -1, 0)[:, :, :, np.newaxis]
    return lc, hc


def make_strided_windows(X, y, series_len, depth, stride, step=1):
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    
    span_needed = (depth - 1) * stride + 1
    n_vols_per_series = series_len - span_needed + 1

    if n_vols_per_series <= 0:
        return np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C))

    X_volumes = []
    y_volumes = []

    for i in range(n_series):
        base = i * series_len
        blockX = X[base : base+series_len]
        blockY = y[base : base+series_len]

        for start_idx in range(0, n_vols_per_series, step):
            indices = np.arange(start_idx, start_idx + span_needed, stride)
            
            if indices[-1] >= series_len:
                continue
                
            X_volumes.append(blockX[indices])
            y_volumes.append(blockY[indices])

    if len(X_volumes) == 0:
        return np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C))
        
    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0)


def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x, y = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x), tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)

        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        sx = tf.reduce_sum(tf.nn.relu(x), [1, 2, 3], keepdims=True) + 1e-12
        sy = tf.reduce_sum(tf.nn.relu(y), [1, 2, 3], keepdims=True) + 1e-12
        sc = tf.random.uniform([], scale_min, scale_max)
        return (x / sx) * sc, (y / sy) * sc
    return map_vol


def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]


# =====================================================
# 4. LR WARMUP CALLBACK
# =====================================================
def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr


# =====================================================
# 5. MAIN SCRIPT
# =====================================================
print(f"--- STARTE TRAINING PUNKT {MY_POINT_IDX} (a={MY_ALPHA:.4f}, b={MY_BETA:.4f}) FÜR SEED {current_seed} ---")

point_dir = SCRATCH_ROOT / f"Point_{MY_POINT_IDX:02d}_a{MY_ALPHA:.4f}_b{MY_BETA:.4f}"
point_dir.mkdir(exist_ok=True)

# -----------------------------------------------------
# FENSTERBILDUNG MIT STRIDES (Trainingsdaten)
# -----------------------------------------------------
suffix_file = "pois_on.hdf5" if USE_POISSON_NOISE else "pois_off.hdf5"
TRAIN_FILE = INTERP_DIR / f"interpolated_training_data_{suffix_file}"
VAL_FILE = ORIGINAL_DATA_DIR / "validation_data.hdf5"

print(f"Lade interpolierte Trainingsdaten: {TRAIN_FILE}")
X_train_raw, y_train_raw = load_split(TRAIN_FILE)

X_train_list = []
y_train_list = []
SELECTED_STRIDES = [1, 2, 4, 6, 12, 24] 

print(f"Generiere Volumina für Strides: {SELECTED_STRIDES}...")
for stride_size in SELECTED_STRIDES:
    if stride_size == 12:
        current_step = 5
    elif stride_size == 24:
        current_step = 4
    else:
        current_step = 6

    X_vol, y_vol = make_strided_windows(
        X_train_raw, 
        y_train_raw, 
        SERIES_LEN_INTERP, 
        DEPTH, 
        stride=stride_size, 
        step=current_step
    )
    
    if len(X_vol) > 0:
        X_train_list.append(X_vol)
        y_train_list.append(y_vol)

# RAM Fix: Rohe Trainingsdaten aus Speicher werfen
del X_train_raw, y_train_raw
gc.collect()

X_train_win = np.concatenate(X_train_list, axis=0)
y_train_win = np.concatenate(y_train_list, axis=0)

# RAM Fix: Listen aus Speicher werfen
del X_train_list, y_train_list
gc.collect()

# -----------------------------------------------------
# FENSTERBILDUNG (Validierungsdaten Original)
# -----------------------------------------------------
print(f"Lade originale Validierungsdaten: {VAL_FILE}")
lc_v, hc_v = load_split(VAL_FILE)
X_val_win, y_val_win = make_strided_windows(lc_v, hc_v, SERIES_LEN_ORIG, DEPTH, stride=1, step=1)

# RAM Fix: Rohe Validierungsdaten aus Speicher werfen
del lc_v, hc_v
gc.collect()


# -----------------------------------------------------
# JOB / SEED EXECUTION
# -----------------------------------------------------
RUN_NAME = f"P{MY_POINT_IDX:02d}_a{MY_ALPHA:.4f}_b{MY_BETA:.4f}_seed{current_seed}"

lock_file = point_dir / f"{RUN_NAME}.lock"
state_file = point_dir / f"{RUN_NAME}__state.json"
csv_file = point_dir / f"{RUN_NAME}_03_metrics.csv"

# Finale Dateipfade fuer die BESTE Epoche
best_keras_file = point_dir / f"{RUN_NAME}_best_model.keras"
best_h5_file = point_dir / f"{RUN_NAME}_best_weights.weights.h5"

# 1) Skip-Logik via state_file
state = read_state_file(state_file)
if state is not None:
    st = state.get("status", "unknown")
    tr = state.get("termination_reason", "unknown")
    if st == "finished" and tr in ["max_epochs_200", "early_stopping", "psnr_safety_net"]:
        print(f"Skipping {RUN_NAME}: bereits erfolgreich abgeschlossen.")
        sys.exit(0)

# 2) Lock atomisch holen
if not acquire_lock(lock_file):
    print(f"Skipping {RUN_NAME}: Lock aktiv (anderer Job trainiert).")
    sys.exit(0)

print(f"\n>>> STARTING TRAINING FOR {RUN_NAME}")

temp_csv = f"temp_{RUN_NAME}.csv"
# Temporaerer Checkpoint fuer die besten Gewichte waehrend des Laufs
temp_checkpoint_file = point_dir / f"{RUN_NAME}_TEMP_best_weights.weights.h5"

try:
    # State auf 'running' setzen
    write_json_atomic(state_file, {
        "status": "running", "run_name": RUN_NAME, "point_idx": MY_POINT_IDX,
        "alpha": MY_ALPHA, "beta": MY_BETA, "seed": current_seed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "termination_reason": "running"
    })

    # Datasets vorbereiten
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train_win, y_train_win))
                .shuffle(len(X_train_win), seed=current_seed)
                .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), num_parallel_calls=AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE).prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val_win, y_val_win))
                .map(augment_and_normalize_3d_per_slice(10000, 10001, 0), num_parallel_calls=AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
                .cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

    model = unet_2d_stacked()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True)

    status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
    term_reason = "crash_or_timeout"

    def check_crash(epoch, logs):
        psnr = logs.get("val_psnr_clipped", 0)
        if psnr > status["best_psnr"]:
            status["best_psnr"] = psnr
            status["drop_cnt"] = 0
        elif epoch >= 10:
            if psnr < (status["best_psnr"] - 4.5) or psnr < 24.0:
                status["drop_cnt"] += 1
            if status["drop_cnt"] >= 3:
                status["aborted"] = True
                status["reason"] = "perf_collapse"
                model.stop_training = True
        touch_lock(lock_file)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(temp_checkpoint_file),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
        tf.keras.callbacks.CSVLogger(temp_csv),
        *tb_callbacks(TB_ROOT / RUN_NAME),
    ]

    model.compile(optimizer=optimizer, loss=get_triple_loss(MY_ALPHA, MY_BETA),
                    metrics=["mae", "mse", mae_clipped, mse_clipped, ssim_clipped, psnr_clipped])

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    if status["aborted"]:
        term_reason = "psnr_safety_net"
    else:
        term_reason = "early_stopping" if len(history.history["loss"]) < EPOCHS else "max_epochs_200"

except Exception as e:
    status["aborted"] = True
    status["reason"] = f"crash_{str(e)[:25]}"
    term_reason = "crash_or_timeout"

# --- FINALISIERUNG DER BESTEN EPOCHE ---
best_eval = {}
best_epoch = None
best_val_loss = None
best_val_psnr = None

if temp_checkpoint_file.exists():
    print(f">>> Lade beste Gewichte zur Speicherung nach: {temp_checkpoint_file.name}")
    model.load_weights(str(temp_checkpoint_file))

    # Doppelte Speicherung der besten Epoche
    model.save(str(best_keras_file)) # Komplettes Modell
    model.save_weights(str(best_h5_file)) # Reine Gewichte

    # Exakte Metriken des BESTEN Zustands evaluieren
    try:
        eval_out = model.evaluate(val_ds, verbose=0, return_dict=True)
        best_eval = {k: float(v) for k, v in eval_out.items()}
        best_val_psnr = best_eval.get("psnr_clipped", None)
        best_val_loss = best_eval.get("loss", None)
    except Exception as e:
        print(f"Warnung: Evaluation fehlgeschlagen: {e}")

# Best-Epoch aus History bestimmen
if history is not None and "val_loss" in history.history:
    best_idx = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_idx + 1
    if best_val_loss is None: best_val_loss = float(history.history["val_loss"][best_idx])
    if best_val_psnr is None: best_val_psnr = float(history.history["val_psnr_clipped"][best_idx])

# finalize_run mit Best-Metriken aufrufen
if history is not None or status["aborted"]:
    final_psnr_for_meta = float(best_val_psnr) if best_val_psnr is not None else float("nan")
    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (192, 240, 5),
                            extra={"alpha": MY_ALPHA, "beta": MY_BETA, "seed": current_seed,
                                    "aborted": status["aborted"], "reason": status["reason"],
                                    "termination_reason": term_reason, "final_psnr": final_psnr_for_meta,
                                    "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                                    "strides": f"1-24"})
    finalize_run(model, history, RUN_NAME, meta, folder_name=str(point_dir))

if os.path.exists(temp_csv):
    shutil.move(temp_csv, csv_file)

# Finalen State schreiben
write_json_atomic(state_file, {
    "status": "finished" if term_reason in ["max_epochs_200", "early_stopping", "psnr_safety_net"] else "incomplete",
    "run_name": RUN_NAME, "finished_at": datetime.now().isoformat(timespec="seconds"),
    "termination_reason": term_reason, "best_epoch": best_epoch, "best_val_psnr": best_val_psnr
})

# Aufräumen
if temp_checkpoint_file.exists(): temp_checkpoint_file.unlink()
if lock_file.exists(): lock_file.unlink()
tf.keras.backend.clear_session(); gc.collect()

print(f"\n--- SEED {current_seed} ERFOLGREICH BEENDET ---")