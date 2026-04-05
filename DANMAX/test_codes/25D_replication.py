#!/usr/bin/env python3

import os
import sys
import random
import gc
import resource
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# 1. KONFIGURATION / PARAMETER
# =====================================================
SEED = 42
EPOCHS = 200
BATCH_SIZE = 8
LR_TARGET = 5e-4
WARMUP_EPOCHS = 10

DEPTH = 5            # Slices im 2.5D Input
SERIES_LEN = 40      # Muss glatt durch N teilbar sein
BASEFILTERS = 64
CROP_SIZE = (256, 256)

# Pfade
BAMBOO_RAW = "../../DATA_DANMAX/2026020508/raw/bamboo/"
BASE_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "test_models"
CACHE_DIR = Path.home() / "scratch" / "DANMAX" / "data_cache"

# =====================================================
# 2. AUTOMATISCHES SETUP & RAM-CHECKER
# =====================================================
def print_peak_ram(label="Aktuell"):
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[RAM-CHECK] Peak RAM Usage ({label}): {peak_kb / (1024 ** 2):.2f} GB\n")

SCRIPT_NAME = Path(__file__).stem
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{SCRIPT_NAME}__{TIMESTAMP}__seed{SEED}"

RUN_DIR = BASE_OUT_DIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# =====================================================
# 3. HELPER, ARCHITEKTUR & METRIKEN
# =====================================================
def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

def conv_block_2d(x, filters):
    ki = "he_normal"
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=ki, use_bias=True)(x)
        x = layers.ReLU()(x)
    return x

def unet_2d_stacked(input_shape):
    inputs = layers.Input(shape=input_shape, name="input")
    c1 = conv_block_2d(inputs, BASEFILTERS);         p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, BASEFILTERS * 2);         p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, BASEFILTERS * 4);         p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, BASEFILTERS * 8);         p4 = layers.MaxPooling2D((2, 2))(c4)
    bn = conv_block_2d(p4, BASEFILTERS * 16)
    u4 = layers.Conv2DTranspose(BASEFILTERS * 8, (2, 2), strides=(2, 2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]);             c5 = conv_block_2d(u4, BASEFILTERS * 8)
    u3 = layers.Conv2DTranspose(BASEFILTERS * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3]);             c6 = conv_block_2d(u3, BASEFILTERS * 4)
    u2 = layers.Conv2DTranspose(BASEFILTERS * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2]);             c7 = conv_block_2d(u2, BASEFILTERS * 2)
    u1 = layers.Conv2DTranspose(BASEFILTERS, (2, 2), strides=(2, 2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1]);             c8 = conv_block_2d(u1, BASEFILTERS)
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(c8)
    return models.Model(inputs, out)

# Metriken arbeiten nativ auf [0, 1]
def mae_true(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_true(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

def psnr_true(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_true(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def mae_ssim_2d_loss(y_true, y_pred, alpha=0.6):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_v = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim_v)

# =====================================================
# 4. DATENVERARBEITUNG & CHUNKED CACHING (70/20/10)
# =====================================================
def load_and_crop_4_quadrants_chunked(base_path, scan_id, start_img, end_img):
    ct_file    = os.path.join(base_path, f"scan-{scan_id:04d}_orca.h5")
    white_file = os.path.join(base_path, f"scan-{scan_id-1:04d}_orca.h5")
    dark_file  = os.path.join(base_path, f"scan-{scan_id-2:04d}_orca.h5")
    data_path  = 'entry/instrument/orca/data'

    with h5py.File(dark_file, 'r') as f_d, h5py.File(white_file, 'r') as f_w, h5py.File(ct_file, 'r') as f_ct:
        m_dark  = np.mean(f_d[data_path][:], axis=0).astype(np.float32)
        m_white = np.mean(f_w[data_path][:], axis=0).astype(np.float32)
        denom = m_white - m_dark; denom[denom < 1e-6] = 1e-6

        ct_data = f_ct[data_path]
        H, W = ct_data.shape[1], ct_data.shape[2]
        cy, cx = H // 2, W // 2
        ch, cw = CROP_SIZE[0], CROP_SIZE[1]

        tl, tr, bl, br = [], [], [], []
        for s in range(start_img, end_img, 100):
            e = min(s + 100, end_img)
            corrected = np.clip((ct_data[s:e].astype(np.float32) - m_dark) / denom, 0, 1)
            tl.append(corrected[:, cy-ch:cy, cx-cw:cx])
            tr.append(corrected[:, cy-ch:cy, cx:cx+cw])
            bl.append(corrected[:, cy:cy+ch, cx-cw:cx])
            br.append(corrected[:, cy:cy+ch, cx:cx+cw])

        return np.concatenate([np.concatenate(tl), np.concatenate(tr), np.concatenate(bl), np.concatenate(br)], axis=0)

def prepare_cache():
    paths = {k: CACHE_DIR / f"{k}.npy" for k in ["t_l", "t_h", "v_l", "v_h", "test_l", "test_h"]}
    if all(p.exists() for p in paths.values()):
        print("Cache gefunden.")
        return paths

    print("Erstelle 70/20/10 Cache (Chunked & Leakage-free)...")
    splits = {"train": (0, 1400), "val": (1400, 1800), "test": (1800, 2000)}

    for mode, (s, e) in splits.items():
        prefix = "t" if mode == "train" else "v" if mode == "val" else "test"
        # Low Count (X)
        data_x = load_and_crop_4_quadrants_chunked(BAMBOO_RAW, 57, s, e)
        np.save(paths[f"{prefix}_l"], data_x.reshape((-1, SERIES_LEN, 256, 256)))
        del data_x; gc.collect()
        # High Count (y)
        data_y = load_and_crop_4_quadrants_chunked(BAMBOO_RAW, 32, s, e)
        np.save(paths[f"{prefix}_h"], data_y.reshape((-1, SERIES_LEN, 256, 256)))
        del data_y; gc.collect()

    return paths

# =====================================================
# 5. DATASET & GENERATOR
# =====================================================
def create_tf_dataset(low_path, high_path, is_training=True):
    X_mmap = np.load(low_path, mmap_mode='r')
    y_mmap = np.load(high_path, mmap_mode='r')

    valid_starts = [[s_idx, w_idx] for s_idx in range(X_mmap.shape[0]) for w_idx in range(SERIES_LEN - DEPTH + 1)]
    valid_starts = np.array(valid_starts, dtype=np.int32)

    def load_window(idx):
        s, w = idx[0].numpy(), idx[1].numpy()
        return X_mmap[s, w:w+DEPTH].copy()[..., np.newaxis], y_mmap[s, w:w+DEPTH].copy()[..., np.newaxis]

    def tf_load_window(idx):
        x, y = tf.py_function(load_window, [idx], [tf.float32, tf.float32])
        x.set_shape([DEPTH, 256, 256, 1])
        y.set_shape([DEPTH, 256, 256, 1])
        return x, y

    def prepare_25d(x, y):
        # Konvertiert [DEPTH, H, W, 1] zu [H, W, DEPTH]
        return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[DEPTH//2]

    def augment_and_scale(x, y):
        if is_training:
            # 1. Flip Left/Right (50%)
            do_flip_lr = tf.random.uniform([]) < 0.5
            x, y = tf.cond(do_flip_lr, lambda: (tf.image.flip_left_right(x), tf.image.flip_left_right(y)), lambda: (x, y))

            # 2. Flip Up/Down (50%)
            do_flip_ud = tf.random.uniform([]) < 0.5
            x, y = tf.cond(do_flip_ud, lambda: (tf.image.flip_up_down(x), tf.image.flip_up_down(y)), lambda: (x, y))

            # 3. Rotation um 90 Grad (25%)
            do_rot = tf.random.uniform([]) < 0.25
            x, y = tf.cond(do_rot, lambda: (tf.image.rot90(x, k=1), tf.image.rot90(y, k=1)), lambda: (x, y))

            # 4. Intensitätsskalierung (0.6 bis 1.0)
            scale = tf.random.uniform([], 0.6, 1.0)
            return x * scale, y * scale
        else:
            # Für Validation/Test: Keine Augmentation, Skalierung = 1.0
            return x, y

    ds = tf.data.Dataset.from_tensor_slices(valid_starts)
    if is_training: ds = ds.shuffle(len(valid_starts), seed=SEED)

    ds = ds.map(tf_load_window, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(prepare_25d, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment_and_scale, num_parallel_calls=tf.data.AUTOTUNE)

    if not is_training: ds = ds.cache()
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 6. TRAINING & EVALUATION
# =====================================================
paths = prepare_cache()
print_peak_ram("Nach Caching")

train_ds = create_tf_dataset(paths["t_l"], paths["t_h"], True)
val_ds   = create_tf_dataset(paths["v_l"], paths["v_h"], False)
test_ds  = create_tf_dataset(paths["test_l"], paths["test_h"], False)

model = unet_2d_stacked((256, 256, DEPTH))
optimizer = tf.keras.optimizers.Adam(LR_TARGET, amsgrad=True)
model.compile(optimizer=optimizer, loss=mae_ssim_2d_loss, metrics=[mae_true, mse_true, psnr_true, ssim_true])

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
    tf.keras.callbacks.ModelCheckpoint(str(RUN_DIR/f"{RUN_NAME}_best_model.keras"), save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(RUN_DIR/f"{RUN_NAME}_metrics.csv"))
]

print(f"Training startet: {RUN_NAME}")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

print("\n--- TEST EVALUATION (100% UNABHÄNGIG) ---")
test_results = model.evaluate(test_ds, return_dict=True)
for k, v in test_results.items(): print(f"  {k}: {v:.4f}")

meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (256, 256, DEPTH), extra={"test_results": test_results})
finalize_run(model, history, RUN_NAME, meta, folder_name=str(RUN_DIR))
print_peak_ram("Final")