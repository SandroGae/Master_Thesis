# cross_val_unet_25d_improved_V2.py
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
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm

# Deine Custom-Module
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# Reproduzierbarkeit
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Konfiguration (Mathematisch analog zum Scan-Skript)
ALPHA_OPTIMAL = 4/6
BETA_OPTIMAL = 3/6
DEPTH = 5
SERIES_LEN_ORIG = 41
WARMUP_EPOCHS = 10
LR_TARGET = 5e-4
BASEFILTERS = 64
BATCH_SIZE = 8
EPOCHS = 200 # Erhöht auf 200 analog zum Scan
AUTOTUNE = tf.data.AUTOTUNE

# Pfade
DATA_ROOT = Path.home() / "data"
ORIG_FILE = DATA_ROOT / "original_data/training_data.hdf5"

# --- HILFSFUNKTIONEN ---

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

# METRIKEN FÜR DIE ANALYSE
def mae_clipped(yt, yp): return tf.reduce_mean(tf.abs(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)))
def mse_clipped(yt, yp): return tf.reduce_mean(tf.square(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)))
def psnr_clipped(yt, yp):
    mse = tf.reduce_mean(tf.square(tf.clip_by_value(yt, 0, 1) - tf.clip_by_value(yp[..., 0:1], 0, 1)), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)
def ssim_clipped(yt, yp): return tf.reduce_mean(tf.image.ssim(tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp[..., 0:1], 0, 1), 1.0))

# ROBUSTE CALLBACK-FUNKTION
def get_training_callbacks(fold_name, ckpt_dir, fold_dir, status_dict, model_ref):
    def check_crash(epoch, logs):
        # WICHTIG: Name muss mit compile-metrics übereinstimmen!
        psnr = logs.get('val_psnr_clipped', 0)
        if psnr > status_dict["best_psnr"]: 
            status_dict["best_psnr"] = psnr
            status_dict["drop_cnt"] = 0
        elif epoch >= 10:
            if psnr < (status_dict["best_psnr"] - 45) or psnr < 0.0:
                status_dict["drop_cnt"] += 1
            if status_dict["drop_cnt"] >= 3:
                status_dict["aborted"] = True
                status_dict["reason"] = "perf_collapse"
                model_ref.stop_training = True

    best_model_file = ckpt_dir / f"{fold_name}_best.keras"

    return [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_file), monitor='val_loss', 
                                           save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
        tf.keras.callbacks.CSVLogger(str(fold_dir / f"{fold_name}_metrics.csv")),
        *tb_callbacks(fold_dir)
    ], best_model_file

# --- ARCHITEKTUR --- (Bleibt gleich, da Softplus gewünscht)
def conv_block_2d(x, filters, dropout_rate=0.1):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, (3,3), padding="same", kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    if dropout_rate > 0: x = layers.Dropout(dropout_rate)(x)
    return x

def unet_2d_stacked(input_shape, base_filters):
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

# --- LOSS ---
def get_probabilistic_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        mu, sigma = y_pred[..., 0:1], y_pred[..., 1:2]
        y_true = tf.cast(y_true, tf.float32)
        mae_nll = (tf.abs(y_true - mu) / sigma) + tf.math.log(sigma)
        mse_nll = (tf.square(y_true - mu) / (2.0 * tf.square(sigma))) + tf.math.log(sigma)
        pixel_loss = (beta * mse_nll) + ((1.0 - beta) * mae_nll)
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, mu, 1.0))
        return (alpha * ssim_loss) + ((1.0 - alpha) * tf.reduce_mean(pixel_loss))
    return loss

# --- DATA LOADING & WINDOWING --- (Analog zum 10-Seeds Scan)
def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        lc, hc = f["low_count/data"][:].astype("float32"), f["high_count/data"][:].astype("float32")
    lc = np.moveaxis(lc, -1, 0)[:, :, :, np.newaxis]
    hc = np.moveaxis(hc, -1, 0)[:, :, :, np.newaxis]
    return lc, hc

def make_stride1_windows(X, y, series_len, depth):
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(X.shape[0] // series_len):
        bx, by = X[i*series_len : (i+1)*series_len], y[i*series_len : (i+1)*series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s+depth]); y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p_flip=0.5):
    def map_vol(x, y):
        x, y = tf.nn.relu(x), tf.nn.relu(y)
        if p_flip > 0:
            flip = tf.random.uniform([], 0, 1) < p_flip
            x, y = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x), tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sx = tf.reduce_sum(x, [1,2,3], keepdims=True) + 1e-12
        sy = tf.reduce_sum(y, [1,2,3], keepdims=True) + 1e-12
        scale = tf.random.uniform([], scale_min, scale_max)
        return (x/sx)*scale, (y/sy)*scale
    return map_vol

# --- VISUALISIERUNG ---
def save_uncertainty_analysis(model, sample_x, fold_id, folder, threshold=0.15):
    prediction = model.predict(sample_x[:1], verbose=0)
    mu, sigma = prediction[0, ..., 0], prediction[0, ..., 1]
    mu_vis = vis_norm(mu, 0.5, 99.5)
    input_vis = vis_norm(sample_x[0, ..., 2], 0.5, 99.5)
    uncertain_mask = np.where(sigma > threshold, 1.0, 0.0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    axes[0].imshow(input_vis, cmap='gray'); axes[0].set_title("Input (Center Slice)")
    axes[1].imshow(mu_vis, cmap='gray'); axes[1].set_title(f"Rekonstruktion mu")
    im_sigma = axes[2].imshow(sigma, cmap='inferno'); axes[2].set_title("Unsicherheit sigma")
    fig.colorbar(im_sigma, ax=axes[2])
    axes[3].imshow(uncertain_mask, cmap='binary'); axes[3].set_title(f"Uncertainty Mask")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder / f"fold_{fold_id}_uncertainty_analysis.png")
    plt.close(fig)

# --- HAUPTABLAUF ---
X_raw, y_raw = load_split(ORIG_FILE)
series_indices = np.arange(len(X_raw) // SERIES_LEN_ORIG)
BASE_NAME, RUN_ID = "confidence_unet_25d_improved_V2", datetime.now().strftime("%Y%m%d-%H%M%S")
TB_ROOT = Path.home() / "data" / "tblogs_unet_3d_simple"

split_idx = int(0.8 * len(series_indices))
manual_split = [(series_indices[:split_idx], series_indices[split_idx:])]

for fold, (train_idx, val_idx) in enumerate(manual_split):
    fold_id = fold + 1
    FOLD_DIR = TB_ROOT / f"{BASE_NAME}_fold{fold_id}_{RUN_ID}" ; FOLD_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = FOLD_DIR / "checkpoints" ; CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def get_data_split(idx):
        X_l, y_l = [], []
        for i in idx:
            s = i * SERIES_LEN_ORIG
            X_l.append(X_raw[s:s+SERIES_LEN_ORIG]); y_l.append(y_raw[s:s+SERIES_LEN_ORIG])
        return np.concatenate(X_l), np.concatenate(y_l)

    X_tr_fold, y_tr_fold = get_data_split(train_idx)
    X_va_fold, y_va_fold = get_data_split(val_idx)

    X_tr_win, y_tr_win = make_stride1_windows(X_tr_fold, y_tr_fold, SERIES_LEN_ORIG, DEPTH)
    X_va_win, y_va_win = make_stride1_windows(X_va_fold, y_va_fold, SERIES_LEN_ORIG, DEPTH)

    model = unet_2d_stacked((192, 240, DEPTH), BASEFILTERS)
    optimizer = tf.keras.optimizers.Adam(LR_TARGET, amsgrad=True, global_clipnorm=1.0)
    
    model.compile(optimizer=optimizer, 
                  loss=get_probabilistic_triple_loss(ALPHA_OPTIMAL, BETA_OPTIMAL), 
                  metrics=[mae_clipped, mse_clipped, ssim_clipped, psnr_clipped])

    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr_win, y_tr_win)).shuffle(len(X_tr_win), seed=SEED)
                .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, 0.5), -1)
                .map(prepare_25d_input, -1).batch(BATCH_SIZE).prefetch(-1))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_va_win, y_va_win))
              .map(augment_and_normalize_3d_per_slice(10000.0, 10000.0, 0), -1)
              .map(prepare_25d_input, -1).cache().batch(BATCH_SIZE).prefetch(-1))

    status = {"best_psnr": -1.0, "drop_cnt": 0, "aborted": False, "reason": "none"}
    callbacks, best_model_path = get_training_callbacks(f"{BASE_NAME}_fold{fold_id}", CKPT_DIR, FOLD_DIR, status, model)

    history = None
    try:
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    except Exception as e:
        status["aborted"] = True ; status["reason"] = f"crash_{str(e)[:20]}"

    # FINALISIERUNG (Analoge Speicherlogik)
    if best_model_path.exists():
        print(f">>> Lade beste Gewichte zur Finalisierung: {best_model_path}")
        model.load_weights(str(best_model_path))

    # Explizite Evaluation für Meta-File
    eval_dict = model.evaluate(val_ds, return_dict=True, verbose=0)
    final_psnr = eval_dict.get("psnr_clipped", status["best_psnr"])

    # Speichern der Gewichte im H5 Format (zusätzlich zum .keras vom Callback)
    best_weights_h5 = FOLD_DIR / f"{BASE_NAME}_fold{fold_id}_best_weights.weights.h5"
    model.save_weights(str(best_weights_h5))

    meta = make_meta_dict(f"{BASE_NAME}_fold{fold_id}", BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (192,240,DEPTH),
                          extra={"aborted": status["aborted"], "reason": status["reason"], 
                                 "final_psnr_eval": float(final_psnr), "alpha": ALPHA_OPTIMAL, "beta": BETA_OPTIMAL})
    
    finalize_run(model, history, f"{BASE_NAME}_fold{fold_id}", meta, folder_name=str(FOLD_DIR))
    
    for sx, sy in val_ds.take(1): save_uncertainty_analysis(model, sx, fold_id, FOLD_DIR)
    
    tf.keras.backend.clear_session()
    del model, train_ds, val_ds, X_tr_win, y_tr_win, X_va_win, y_va_win

print(f"\nTraining abgeschlossen.")