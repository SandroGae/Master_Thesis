#!/usr/bin/env python3

import os
import sys
import random
import gc
import shutil
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# HDF5 Locking für HPCs deaktivieren, um den Errno 11 zu vermeiden
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# 1. SETUP & ARGUMENT PARSING
# =====================================================
# Definiere die 3 gewünschten Punkte als Liste von Dictionaries
# Index 0: Punkt 2, Index 1: Punkt 14, Index 2: Punkt 23
CONFIGS = [
    {"point_id": 2,  "alpha": 0.0, "beta": 2/6},
    {"point_id": 14, "alpha": 2/6, "beta": 0.0},
    {"point_id": 23, "alpha": 3/6, "beta": 2/6}
]

# Wir erzeugen 30 Runs (3 Punkte * 10 Seeds). 
# Der run_idx geht von 0 bis 29.
parser = argparse.ArgumentParser()
parser.add_argument("--run_idx", type=int, required=True, help="Index von 0 bis 29 (3 Punkte * 10 Seeds)")
args = parser.parse_args()

run_idx = args.run_idx
if not (0 <= run_idx < 30):
    print(f"Fehler: run_idx {run_idx} ist außerhalb des gültigen Bereichs (0-29).")
    sys.exit(1)

# Aus dem run_idx berechnen wir, welcher Punkt und welcher Seed dran ist
config_idx = run_idx // 10
seed_offset = run_idx % 10

CURRENT_CONFIG = CONFIGS[config_idx]
POINT_ID = CURRENT_CONFIG["point_id"]
ALPHA_OPTIMAL = CURRENT_CONFIG["alpha"]
BETA_OPTIMAL = CURRENT_CONFIG["beta"]
CURRENT_SEED = 42 + seed_offset # Seeds 42 bis 51

# Reproduzierbarkeit für diesen spezifischen Seed setzen
os.environ['PYTHONHASHSEED'] = str(CURRENT_SEED)
random.seed(CURRENT_SEED)
np.random.seed(CURRENT_SEED)
tf.random.set_seed(CURRENT_SEED)
tf.config.experimental.enable_op_determinism()

# Konfiguration
DEPTH = 5
SERIES_LEN_ORIG = 41
BATCH_SIZE = 8
EPOCHS = 250 # Auf 250 erhöht
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10
BASEFILTERS = 64
AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT = Path.home() / "data" / "original_data"
# Eigener Ordner für dieses spezifische Experiment
SCRATCH_ROOT = Path.home() / "scratch" / "Final_3_Points_Experiment"
TB_ROOT = SCRATCH_ROOT / "tensorboard_logs"
MODEL_DIR = SCRATCH_ROOT / "models"

for d in [SCRATCH_ROOT, TB_ROOT, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e: print(f"Warnung: {e}")

# --- HILFSFUNKTIONEN VISUALISIERUNG ---
def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def save_uncertainty_analysis(model, sample_x, seed, folder, threshold=0.15):
    prediction = model.predict(sample_x[:1], verbose=0)
    mu, sigma = prediction[0, ..., 0], prediction[0, ..., 1]
    
    mu_vis = vis_norm(mu)
    input_vis = vis_norm(sample_x[0, ..., 2])
    uncertain_mask = np.where(sigma > threshold, 1.0, 0.0)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    axes[0].imshow(input_vis, cmap='gray'); axes[0].set_title("Input (Center Slice)")
    axes[1].imshow(mu_vis, cmap='gray'); axes[1].set_title("Rekonstruktion (mu)")
    im_sigma = axes[2].imshow(sigma, cmap='inferno'); axes[2].set_title("Unsicherheit (sigma)")
    fig.colorbar(im_sigma, ax=axes[2])
    axes[3].imshow(uncertain_mask, cmap='binary'); axes[3].set_title(f"Uncertainty Mask (>{threshold})")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder / f"seed_{seed}_uncertainty_analysis.png")
    plt.close(fig)

# =====================================================
# 2. METRIKEN & LOSS
# =====================================================
def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS: return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

def mae_raw(yt, yp): return tf.reduce_mean(tf.abs(yt - yp[..., 0:1]))
def mse_raw(yt, yp): return tf.reduce_mean(tf.square(yt - yp[..., 0:1]))
def ssim_raw(yt, yp): return tf.reduce_mean(tf.image.ssim(yt, yp[..., 0:1], max_val=1.0))
def psnr_raw(yt, yp):
    mse = tf.reduce_mean(tf.square(yt - yp[..., 0:1]), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def mae_clipped(yt, yp): return tf.reduce_mean(tf.abs(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)))
def mse_clipped(yt, yp): return tf.reduce_mean(tf.square(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)))
def psnr_clipped(yt, yp):
    mse = tf.reduce_mean(tf.square(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)
def ssim_clipped(yt, yp): return tf.reduce_mean(tf.image.ssim(tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp[..., 0:1], 0, 1), 1.0))
def avg_sigma(yt, yp): return tf.reduce_mean(yp[..., 1:2])

def get_probabilistic_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        mu = y_pred[..., 0:1]
        sigma = y_pred[..., 1:2]
        sigma = tf.maximum(sigma, 1e-6)
        
        y_true = tf.cast(y_true, tf.float32)
        mae_nll = (tf.abs(y_true - mu) / sigma) + tf.math.log(sigma)
        mse_nll = (tf.square(y_true - mu) / (2.0 * tf.square(sigma))) + tf.math.log(sigma)
        pixel_loss = (beta * mse_nll) + ((1.0 - beta) * mae_nll)
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, mu, 1.0))
        return (alpha * ssim_loss) + ((1.0 - alpha) * tf.reduce_mean(pixel_loss))
    return loss

# =====================================================
# 3. ARCHITEKTUR & DATA UTILS
# =====================================================
def conv_block_2d(x, filters, dropout_rate=0.1):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, (3,3), padding="same", kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    if dropout_rate > 0: x = layers.Dropout(dropout_rate)(x)
    return x

def unet_2d_stacked(input_shape=(192, 240, 5), base_filters=64):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, base_filters) ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2) ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4) ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8) ; p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, base_filters * 16)
    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]) ; c5 = conv_block_2d(u4, base_filters * 8, 0.0)
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]) ; c6 = conv_block_2d(u3, base_filters * 4, 0.0)
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]) ; c7 = conv_block_2d(u2, base_filters * 2, 0.0)
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]) ; c8 = conv_block_2d(u1, base_filters, 0.0)
    
    x = layers.Conv2D(filters=2, kernel_size=(1, 1), activation="linear", name="output_raw")(c8) 
    mu = layers.Activation("sigmoid", name="mu_output")(x[..., 0:1]) 
    sigma = layers.Lambda(lambda t: tf.math.softplus(t) + 1e-3, name="sigma_output")(x[..., 1:2]) 
    out = layers.Concatenate()([mu, sigma])
    return models.Model(inputs, out)

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        lc = f["low_count/data"][:].astype("float32")
        hc = f["high_count/data"][:].astype("float32")
    lc = np.moveaxis(lc, -1, 0)[:, :, :, np.newaxis]
    hc = np.moveaxis(hc, -1, 0)[:, :, :, np.newaxis]
    return lc, hc

def make_stride1_windows(X, y, series_len, depth):
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(X.shape[0] // series_len):
        bx = X[i * series_len: (i + 1) * series_len]
        by = y[i * series_len: (i + 1) * series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s + depth]); y_v.append(by[s:s + depth])
    return np.stack(X_v), np.stack(y_v)

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p_flip=0.5):
    def map_vol(x, y):
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        if p_flip > 0:
            flip = tf.random.uniform([], 0, 1) < p_flip
            x, y = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x), tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sx = tf.reduce_sum(x, [1, 2, 3], keepdims=True) + 1e-12
        sy = tf.reduce_sum(y, [1, 2, 3], keepdims=True) + 1e-12
        scale = tf.random.uniform([], scale_min, scale_max)
        return (x / sx) * scale, (y / sy) * scale
    return map_vol

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

# =====================================================
# 4. TRAINING EXECUTION
# =====================================================
print(f"\n{'='*80}")
print(f">>> STARTING JOB {run_idx+1}/30")
print(f">>> POINT {POINT_ID}: a={ALPHA_OPTIMAL:.4f}, b={BETA_OPTIMAL:.4f} | SEED {CURRENT_SEED}")
print(f"{'='*80}\n")

# Wir haben die Schleife entfernt. Dieser Job macht exakt EINEN Seed!
RUN_NAME = f"P{POINT_ID:02d}_a{ALPHA_OPTIMAL:.4f}_b{BETA_OPTIMAL:.4f}_seed{CURRENT_SEED}"
SEED_DIR = MODEL_DIR / RUN_NAME
SEED_DIR.mkdir(parents=True, exist_ok=True)

csv_file = SEED_DIR / f"{RUN_NAME}_metrics.csv"
best_keras_file = SEED_DIR / f"{RUN_NAME}_best_model.keras"
best_h5_file = SEED_DIR / f"{RUN_NAME}_best_weights.weights.h5"

# Lock und State Mechanismus (für Array Jobs immer noch nützlich, falls man denselben Job zweimal startet)
lock_file = SEED_DIR / f"{RUN_NAME}.lock"
state_file = SEED_DIR / f"{RUN_NAME}__state.json"

if state_file.exists():
    try:
        with open(state_file, "r") as f: state = json.load(f)
        if state.get("status") == "finished":
            print(f"Skipping {RUN_NAME}: bereits erfolgreich abgeschlossen.")
            sys.exit(0)
    except Exception: pass

# Lockfile checken
if lock_file.exists():
    try:
        age = time.time() - lock_file.stat().st_mtime
        if age < 2 * 60 * 60: # jünger als 2h
            print(f"Skipping {RUN_NAME}: Lock aktiv (Job läuft vermutlich woanders).")
            sys.exit(0)
        else: lock_file.unlink(missing_ok=True)
    except Exception: pass

# Lock setzen
lock_file.touch()

# Laden der Daten erst hier, falls der Job geskippt wird, sparen wir RAM und IO
lc_t, hc_t = load_split(DATA_ROOT / "training_data.hdf5")
X_train_win, y_train_win = make_stride1_windows(lc_t, hc_t, SERIES_LEN_ORIG, DEPTH)

lc_v, hc_v = load_split(DATA_ROOT / "validation_data.hdf5")
X_val_win, y_val_win = make_stride1_windows(lc_v, hc_v, SERIES_LEN_ORIG, DEPTH)

train_ds = (tf.data.Dataset.from_tensor_slices((X_train_win, y_train_win))
            .shuffle(len(X_train_win), seed=CURRENT_SEED)
            .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, 0.5), num_parallel_calls=AUTOTUNE)
            .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val_win, y_val_win))
          .map(augment_and_normalize_3d_per_slice(10000.0, 10000.0, 0), num_parallel_calls=AUTOTUNE)
          .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
          .cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

model = unet_2d_stacked((192, 240, DEPTH), BASEFILTERS)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_TARGET, amsgrad=True, global_clipnorm=1.0)

model.compile(
    optimizer=optimizer, 
    loss=get_probabilistic_triple_loss(ALPHA_OPTIMAL, BETA_OPTIMAL), 
    metrics=[mae_clipped, mse_clipped, ssim_clipped, psnr_clipped, mae_raw, mse_raw, ssim_raw, psnr_raw, avg_sigma]
)

status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
term_reason = "crash_or_timeout"

def check_crash(epoch, logs):
    psnr = logs.get('val_psnr_clipped', 0)
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
    lock_file.touch()

temp_checkpoint_file = SEED_DIR / f"{RUN_NAME}_TEMP.weights.h5"

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(temp_checkpoint_file), monitor='val_loss', 
        save_best_only=True, save_weights_only=True, mode='min', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
    tf.keras.callbacks.CSVLogger(str(csv_file)),
    *tb_callbacks(TB_ROOT / RUN_NAME)
]

history = None
try:
    with open(state_file, "w") as f: json.dump({"status": "running"}, f)

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    
    if status["aborted"]:
        term_reason = "psnr_safety_net"
    else:
        term_reason = "early_stopping" if len(history.history["loss"]) < EPOCHS else "max_epochs_250"
except Exception as e:
    status["aborted"] = True
    status["reason"] = f"crash_{str(e)[:20]}"
    term_reason = "crash"

# --- FINALISIERUNG ---
best_epoch = None
best_val_loss = None
best_val_psnr = None

if temp_checkpoint_file.exists():
    print(f">>> Lade beste Gewichte zur Speicherung nach: {temp_checkpoint_file.name}")
    try:
        model.load_weights(str(temp_checkpoint_file))
    except Exception as e:
        print(f"Warnung: Konnte TEMP Gewichte nicht laden: {e}. Versuche kurzen Sleep...")
        time.sleep(5)
        model.load_weights(str(temp_checkpoint_file))
    
    model.save(str(best_keras_file))
    model.save_weights(str(best_h5_file))
    
    try:
        eval_out = model.evaluate(val_ds, verbose=0, return_dict=True)
        best_val_psnr = eval_out.get("psnr_clipped", status["best_psnr"])
        best_val_loss = eval_out.get("loss", None)
    except Exception as e:
        print(f"Warnung: Evaluation fehlgeschlagen: {e}")

if history is not None and "val_loss" in history.history:
    best_idx = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_idx + 1
    if best_val_loss is None: best_val_loss = float(history.history["val_loss"][best_idx])
    if best_val_psnr is None: best_val_psnr = float(history.history["val_psnr_clipped"][best_idx])

final_psnr_for_meta = float(best_val_psnr) if best_val_psnr is not None else float("nan")

meta = make_meta_dict(
    RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (192, 240, DEPTH),
    extra={
        "aborted": status["aborted"], 
        "reason": status["reason"], 
        "termination_reason": term_reason,
        "final_psnr_eval": final_psnr_for_meta, 
        "alpha": ALPHA_OPTIMAL, 
        "beta": BETA_OPTIMAL,
        "seed": CURRENT_SEED,
        "point_id": POINT_ID,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }
)

finalize_run(model, history, RUN_NAME, meta, folder_name=str(SEED_DIR))

for sx, sy in val_ds.take(1): 
    save_uncertainty_analysis(model, sx, CURRENT_SEED, SEED_DIR)

with open(state_file, "w") as f: 
    json.dump({"status": "finished" if term_reason != "crash" else "crash", "reason": term_reason}, f)

if temp_checkpoint_file.exists(): temp_checkpoint_file.unlink()
if lock_file.exists(): lock_file.unlink()

tf.keras.backend.clear_session()
gc.collect()

print(f"\n--- TRAINING ABGESCHLOSSEN FÜR RUN_IDX {run_idx} ---")