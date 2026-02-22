# cross_val_unet_25d_improved_V2.py
#!/usr/bin/env python3

import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg') # Deaktiviert die GUI-Anforderung für Headless-Server
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from tqdm import tqdm

from unet_3d_simple_checkpoints import make_epoch_ckpt_callback, finalize_run, make_meta_dict
from tb_utils import make_run_dir, tb_callbacks

# Reproduzierbarkeit
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Konfiguration
ALPHA_OPTIMAL = 0.0
BETA_OPTIMAL = 0.0
DEPTH = 5
SERIES_LEN_ORIG = 41
BASEFILTERS = 64
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# Pfade
DATA_ROOT = Path.home() / "data"
ORIG_FILE = DATA_ROOT / "original_data/training_data.hdf5"

# --- FUNKTIONEN ---

def conv_block_2d(x, filters, dropout_rate, kernel_size, padding):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def unet_2d_stacked(input_shape, base_filters):
    inputs = layers.Input(shape=input_shape, name="input")
    
    c1 = conv_block_2d(inputs, base_filters, 0.1, (3,3), "same") ; p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, base_filters * 2, 0.1, (3,3), "same") ; p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, base_filters * 4, 0.1, (3,3), "same") ; p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, base_filters * 8, 0.1, (3,3), "same") ; p4 = layers.MaxPooling2D((2, 2))(c4)
    
    bn = conv_block_2d(p4, base_filters * 16, 0.1, (3,3), "same")

    u4 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]) ; c5 = conv_block_2d(u4, base_filters * 8, 0.0, (3,3), "same")
    u3 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]) ; c6 = conv_block_2d(u3, base_filters * 4, 0.0, (3,3), "same")
    u2 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]) ; c7 = conv_block_2d(u2, base_filters * 2, 0.0, (3,3), "same")
    u1 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]) ; c8 = conv_block_2d(u1, base_filters, 0.0, (3,3), "same")
    
    x = layers.Conv2D(filters=2, kernel_size=(1, 1), activation="linear", name="output_raw")(c8) 

    mu = layers.Activation("sigmoid", name="mu_output")(x[..., 0:1]) 
    sigma = layers.Lambda(lambda t: tf.math.softplus(t) + 1e-6, name="sigma_output")(x[..., 1:2]) 
    
    out = layers.Concatenate()([mu, sigma])
    return models.Model(inputs, out, name="unet_25d_stacked")

def load_split(h5_path):
    with h5py.File(h5_path, "r") as f:
        low_ds = f["low_count/data"]
        high_ds = f["high_count/data"]
        num_imgs = low_ds.shape[-1]
        h, w = low_ds.shape[0], low_ds.shape[1]
        low_count = np.empty((num_imgs, h, w, 1), dtype=np.float32)
        high_count = np.empty((num_imgs, h, w, 1), dtype=np.float32)
        print(f"Lade {h5_path}...")
        pbar = tqdm(total=num_imgs, unit="Bilder", desc="RAM Loading")
        chunk_size = 100
        for start in range(0, num_imgs, chunk_size):
            end = min(start + chunk_size, num_imgs)
            low_count[start:end, ..., 0] = np.moveaxis(low_ds[..., start:end], -1, 0)
            high_count[start:end, ..., 0] = np.moveaxis(high_ds[..., start:end], -1, 0)
            current_gb = (low_count.nbytes + high_count.nbytes) / (1024**3)
            pbar.set_postfix({"RAM": f"{current_gb:.2f} GB"})
            pbar.update(end - start)
        pbar.close()
    return low_count, high_count

def make_strided_windows(X, y, series_len, depth, stride, step=1):
    N, H, W, C = X.shape
    n_series = N // series_len
    span_needed = (depth - 1) * stride + 1
    n_vols_per_series = series_len - span_needed + 1
    if n_vols_per_series <= 0: return np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C))
    X_vols, y_vols = [], []
    for i in range(n_series):
        base = i * series_len
        bX, bY = X[base:base+series_len], y[base:base+series_len]
        for start_idx in range(0, n_vols_per_series, step):
            indices = np.arange(start_idx, start_idx + span_needed, stride)
            if indices[-1] >= series_len: continue
            X_vols.append(bX[indices])
            y_vols.append(bY[indices])
    return (np.stack(X_vols, axis=0), np.stack(y_vols, axis=0)) if X_vols else (np.empty((0, depth, H, W, C)), np.empty((0, depth, H, W, C)))

def shuffle_initial(X, y, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]

def prepare_25d_input(x, y):
    x = tf.squeeze(x, axis=-1)
    x = tf.transpose(x, [1, 2, 0])
    idx = tf.shape(y)[0] // 2
    return x, y[idx]

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p_flip=0.5):
    def map_volume(x, y):
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)
        if p_flip > 0:
            flip = tf.random.uniform([], 0, 1) < p_flip
            x = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x)
            y = tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[1, 2, 3], keepdims=True) + 1e-12
        x, y = x / sum_x, y / sum_y
        scale = tf.random.uniform([], scale_min, scale_max)
        return x * scale, y * scale
    return map_volume

def get_probabilistic_triple_loss(alpha, beta):
    def loss(y_true, y_pred):
        mu = y_pred[..., 0:1]
        sigma = y_pred[..., 1:2]
        y_true = tf.cast(y_true, tf.float32)
        abs_error = tf.abs(y_true - mu)
        sq_error = tf.square(y_true - mu)
        mae_nll = (abs_error / sigma) + tf.math.log(sigma)
        mse_nll = (sq_error / (2.0 * tf.square(sigma))) + tf.math.log(sigma)
        pixel_loss = (beta * mse_nll) + ((1.0 - beta) * mae_nll)
        ssim_val = tf.image.ssim(y_true, mu, max_val=1.0)
        ssim_loss = 1.0 - tf.reduce_mean(ssim_val)
        return (alpha * ssim_loss) + ((1.0 - alpha) * tf.reduce_mean(pixel_loss))
    return loss

def mae_center(y_true, y_pred):
    y_p = tf.clip_by_value(y_pred[..., 0:1], 0.0, 1.0)
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_t - y_p))

def mse_center(y_true, y_pred):
    y_p = tf.clip_by_value(y_pred[..., 0:1], 0.0, 1.0)
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    return tf.reduce_mean(tf.math.squared_difference(y_t, y_p))

def psnr_center(y_true, y_pred):
    y_p = tf.clip_by_value(y_pred[..., 0:1], 0.0, 1.0)
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_t, y_p), axis=(1, 2, 3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(y_true, y_pred):
    y_p = tf.clip_by_value(y_pred[..., 0:1], 0.0, 1.0)
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))

def save_uncertainty_analysis(model, sample_x, fold_id, folder, threshold):
    prediction = model.predict(sample_x[:1])
    mu = prediction[0, ..., 0]
    sigma = prediction[0, ..., 1]
    uncertain_mask = np.where(sigma > threshold, 1.0, 0.0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(sample_x[0, ..., 2], cmap='gray')
    axes[0].set_title("Input (Center Slice)")
    axes[1].imshow(mu, cmap='gray')
    axes[1].set_title(f"Rekonstruktion mu (Fold {fold_id})")
    im_sigma = axes[2].imshow(sigma, cmap='inferno')
    axes[2].set_title("Unsicherheit sigma")
    fig.colorbar(im_sigma, ax=axes[2])
    axes[3].imshow(uncertain_mask, cmap='binary')
    axes[3].set_title(f"Uncertainty Mask (> {threshold})")
    plt.tight_layout()
    plt.savefig(folder / f"fold_{fold_id}_deep_analysis.png")
    plt.close(fig)

# --- HAUPTABLAUF ---

print(f"Lade Original-Trainingsdaten: {ORIG_FILE}")
X_raw, y_raw = load_split(ORIG_FILE)

num_series = len(X_raw) // SERIES_LEN_ORIG
series_indices = np.arange(num_series)

BASE_NAME = "confidence_no_interp_unet_25d_improved_V2"
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
TB_ROOT = Path.home() / "data" / "tblogs_unet_3d_simple"
all_fold_scores = []

# kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Berechne einen festen Split (80/20)
split_idx = int(0.8 * len(series_indices))
manual_split = [(series_indices[:split_idx], series_indices[split_idx:])]

# Kommentiere die originale Schleife aus und nutze die neue:
# for fold, (train_idx, val_idx) in enumerate(kf.split(series_indices)):
for fold, (train_idx, val_idx) in enumerate(manual_split): 
    fold_id = fold + 1
    # Ab hier bleibt alles wie es ist, inklusive der Einrückung!
    print(f"\n\n{'='*40}\nSTARTE SINGLE RUN (80/20 Split)\n{'='*40}")
    
    FOLD_NAME = f"{BASE_NAME}_fold{fold_id}_{RUN_ID}"
    FOLD_DIR = TB_ROOT / FOLD_NAME
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    def get_data_split(indices):
        X_l, y_l = [], []
        for i in indices:
            start = i * SERIES_LEN_ORIG
            X_l.append(X_raw[start : start + SERIES_LEN_ORIG])
            y_l.append(y_raw[start : start + SERIES_LEN_ORIG])
        return np.concatenate(X_l), np.concatenate(y_l)

    X_tr_fold, y_tr_fold = get_data_split(train_idx)
    X_va_fold, y_va_fold = get_data_split(val_idx)

    print(f"Generiere Volumina für Fold {fold_id}...")
    X_tr_win_list, y_tr_win_list = [], []
    # Strides angepasst auf SERIES_LEN=41 (Maximaler Stride bei Depth 5 ist 10)
    for s in [1, 2, 4, 8]:
        Xw, yw = make_strided_windows(X_tr_fold, y_tr_fold, SERIES_LEN_ORIG, DEPTH, stride=s, step=1)
        if len(Xw) > 0:
            X_tr_win_list.append(Xw.astype(np.float32))
            y_tr_win_list.append(yw.astype(np.float32))
    
    X_tr_win = np.concatenate(X_tr_win_list, axis=0)
    y_tr_win = np.concatenate(y_tr_win_list, axis=0)

    X_va_win, y_va_win = make_strided_windows(X_va_fold, y_va_fold, SERIES_LEN_ORIG, DEPTH, stride=1)

    X_tr_win, y_tr_win = shuffle_initial(X_tr_win, y_tr_win, SEED)
    X_va_win, y_va_win = shuffle_initial(X_va_win, y_va_win, SEED)

    model = unet_2d_stacked(input_shape=(192, 240, DEPTH), base_filters=BASEFILTERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)

    model.compile(
        optimizer=optimizer, 
        loss=get_probabilistic_triple_loss(ALPHA_OPTIMAL, BETA_OPTIMAL), 
        metrics=[mae_center, mse_center, psnr_center, ssim_center]
    )

    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr_win, y_tr_win))
                .shuffle(len(X_tr_win), seed=SEED)
                .map(augment_and_normalize_3d_per_slice(5000.0, 15000.0, p_flip=0.5), num_parallel_calls=AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_va_win, y_va_win))
              .map(augment_and_normalize_3d_per_slice(10000.0, 10000.0, p_flip=0.0), num_parallel_calls=AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=AUTOTUNE)
              .cache()
              .batch(BATCH_SIZE)
              .prefetch(AUTOTUNE))
    
    fold_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        make_epoch_ckpt_callback(FOLD_NAME),
        tf.keras.callbacks.CSVLogger(str(FOLD_DIR / f"{FOLD_NAME}.csv")),
        *tb_callbacks(FOLD_DIR)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=fold_callbacks, verbose=2)

    all_fold_scores.append(min(history.history['val_mae_center']))
    meta = make_meta_dict(FOLD_NAME, BATCH_SIZE, 100, optimizer, 5e-4, (192,240,DEPTH))
    meta["scaling_factor"] = "random_5k_15k"
    meta["augmentation"] = "horizontal_flip_p05"
    meta["uncertainty_type"] = "aleatoric_laplace_and_epistemic_dropout"
    
    finalize_run(model, history, FOLD_NAME, meta)
    for sample_x, sample_y in val_ds.take(1):
        save_uncertainty_analysis(model, sample_x, fold_id, FOLD_DIR, threshold=0.15)

    tf.keras.backend.clear_session()
    del model, train_ds, val_ds, X_tr_win, y_tr_win, X_va_win, y_va_win

print(f"\nK-Fold abgeschlossen. Durchschnittlicher MAE: {np.mean(all_fold_scores):.6f}")