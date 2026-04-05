#!/usr/bin/env python3

import os
import sys
import random
import gc
import resource
import argparse
import json
import time
import socket
from datetime import datetime
from pathlib import Path

# TF-Import nach ENV ist sauberer
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Deine Custom-Module
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# 1. SETUP & ARGUMENT PARSING
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True, help="Index 0-3 für die 4 Jobs")
args = parser.parse_args()

MATERIALS = ["bamboo", "carbon_fiber", "glass_fiber", "chicken_liver"]
SEEDS = [42]  # WICHTIG: Muss eine Liste sein!

job_configs = []
for s in SEEDS:
    for mat in MATERIALS:
        job_configs.append({"point": "P14", "material": mat, "seed": s})

if args.task_id < 0 or args.task_id >= len(job_configs):
    print(f"❌ Ungültige task_id {args.task_id}. Erlaubt sind 0 bis {len(job_configs)-1}.")
    sys.exit(1)

current_config = job_configs[args.task_id]
CURRENT_MATERIAL = current_config["material"]
MY_SEED = current_config["seed"]

# =====================================================
# 2. KONFIGURATION / PARAMETER
# =====================================================
EPOCHS = 200
BATCH_SIZE = 32  # Auf 16 gesetzt für stabile Gradienten bei 256x256
LR_TARGET = 5e-4
WARMUP_EPOCHS = 3

DEPTH = 5            # Slices im 2.5D Input
SERIES_LEN = 40      # Muss glatt durch N teilbar sein
BASEFILTERS = 64
CROP_SIZE = (256, 256)

# Puffer für Translation (Wird nur noch für das Training verwendet!)
LOAD_CROP_SIZE = (288, 288)

# MULTI-DATASET CONFIG (Scan IDs für Low Count / High Count)
DATASETS = {
    "bamboo":        {"gt_id": 32,  "lc_id": 57},  
    "carbon_fiber":  {"gt_id": 60,  "lc_id": 84},  
    "glass_fiber":   {"gt_id": 87,  "lc_id": 111}, 
    "chicken_liver": {"gt_id": 114, "lc_id": 138}  
}

# Pfade
RAW_BASE_DIR = Path("/scratch/sgaell/DATA_DANMAX/2026020508/raw")
BASE_OUT_DIR = Path.home() / "scratch" / "DANMAX" / "test_models"

# =====================================================
# 3. AUTOMATISCHES SETUP & RAM-CHECKER
# =====================================================
def print_peak_ram(label="Aktuell"):
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[RAM-CHECK] Peak RAM Usage ({label}): {peak_kb / (1024 ** 2):.2f} GB\n")

SCRIPT_NAME = Path(__file__).stem
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{SCRIPT_NAME}__{CURRENT_MATERIAL}__{TIMESTAMP}__seed{MY_SEED}"

RUN_DIR = BASE_OUT_DIR / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# WICHTIG: Cache pro Material trennen!
CACHE_DIR = Path.home() / "scratch" / "DANMAX" / "data_cache" / f"{CURRENT_MATERIAL}_{LOAD_CROP_SIZE[0]}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['PYTHONHASHSEED'] = str(MY_SEED)
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.random.set_seed(MY_SEED)
tf.config.experimental.enable_op_determinism()

# =====================================================
# 4. HELPER, ARCHITEKTUR & METRIKEN
# =====================================================
def lr_warmup_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return LR_TARGET * (epoch + 1) / WARMUP_EPOCHS
    return lr

def conv_block_2d(x, filters):
    ki = "he_normal"
    for _ in range(3):
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
    # WICHTIG: Sigmoid bleibt für [0, 1] Bounds!
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(c8)
    return models.Model(inputs, out)

def mae_true(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def mse_true(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

def psnr_true(y_true, y_pred):
    mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_true(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        yt = tf.cast(yt, tf.float32)
        yp = tf.cast(yp, tf.float32)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

p14_loss = get_triple_loss(alpha=2/6, beta=0)

# =====================================================
# 5. DATENVERARBEITUNG & CHUNKED CACHING (70/20/10)
# =====================================================
def load_and_crop_4_quadrants_chunked(base_path, scan_id, start_img, end_img, crop_dim):
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

        ch, cw = crop_dim[0], crop_dim[1]

        tl, tr, bl, br = [], [], [], []
        for s in range(start_img, end_img, 100):
            e = min(s + 100, end_img)
            # WICHTIG: Clipping auf exakt 0, 1 passend zur Sigmoid!
            corrected = np.clip((ct_data[s:e].astype(np.float32) - m_dark) / denom, 0, 1)
            tl.append(corrected[:, cy-ch:cy, cx-cw:cx])
            tr.append(corrected[:, cy-ch:cy, cx:cx+cw])
            bl.append(corrected[:, cy:cy+ch, cx-cw:cx])
            br.append(corrected[:, cy:cy+ch, cx:cx+cw])

        return np.concatenate([np.concatenate(tl), np.concatenate(tr), np.concatenate(bl), np.concatenate(br)], axis=0)

def prepare_cache():
    paths = {k: CACHE_DIR / f"{k}.npy" for k in ["t_l", "t_h", "v_l", "v_h", "test_l", "test_h"]}
    if all(p.exists() for p in paths.values()):
        print(f"Cache für {CURRENT_MATERIAL} gefunden.")
        return paths

    print(f"Erstelle 70/20/10 Cache für {CURRENT_MATERIAL}...")
    splits = {"train": (0, 1400), "val": (1400, 1800), "test": (1800, 2000)}
    
    mat_path = RAW_BASE_DIR / CURRENT_MATERIAL
    ids = DATASETS[CURRENT_MATERIAL]

    for mode, (s, e) in splits.items():
        prefix = "t" if mode == "train" else "v" if mode == "val" else "test"
        # Train bekommt 288x288 Puffer, Val/Test exakt 256x256 (Kein Spatial Shift mehr!)
        c_size = LOAD_CROP_SIZE if mode == "train" else CROP_SIZE
        
        data_x = load_and_crop_4_quadrants_chunked(mat_path, ids["lc_id"], s, e, c_size)
        np.save(paths[f"{prefix}_l"], data_x.reshape((-1, SERIES_LEN, c_size[0], c_size[1])))
        del data_x; gc.collect()
        
        data_y = load_and_crop_4_quadrants_chunked(mat_path, ids["gt_id"], s, e, c_size)
        np.save(paths[f"{prefix}_h"], data_y.reshape((-1, SERIES_LEN, c_size[0], c_size[1])))
        del data_y; gc.collect()

    return paths

# =====================================================
# 6. DATASET & GENERATOR
# =====================================================
def create_tf_dataset(low_path, high_path, is_training=True):
    X_mmap = np.load(low_path, mmap_mode='r')
    y_mmap = np.load(high_path, mmap_mode='r')

    valid_starts = [[s_idx, w_idx] for s_idx in range(X_mmap.shape[0]) for w_idx in range(SERIES_LEN - DEPTH + 1)]
    valid_starts = np.array(valid_starts, dtype=np.int32)
    
    # Dynamische Größe abhängig davon, ob Training (288) oder Val/Test (256)
    current_size = LOAD_CROP_SIZE if is_training else CROP_SIZE

    def load_window(idx):
        s, w = idx[0].numpy(), idx[1].numpy()
        return X_mmap[s, w:w+DEPTH].copy()[..., np.newaxis], y_mmap[s, w:w+DEPTH].copy()[..., np.newaxis]

    def tf_load_window(idx):
        x, y = tf.py_function(load_window, [idx], [tf.float32, tf.float32])
        x.set_shape([DEPTH, current_size[0], current_size[1], 1])
        y.set_shape([DEPTH, current_size[0], current_size[1], 1])
        return x, y

    def prepare_25d(x, y):
        return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[DEPTH//2]

    def augment_and_scale(x, y):
        if is_training:
            xy = tf.concat([x, y], axis=-1)

            # 1. Translation durch Random Crop von 288x288 auf 256x256
            xy = tf.image.random_crop(xy, size=[CROP_SIZE[0], CROP_SIZE[1], DEPTH + 1])

            # 2. Native Flips & Rotation (pro Bild individuell!)
            do_flip_lr = tf.random.uniform([]) < 0.5
            xy = tf.cond(do_flip_lr, lambda: tf.image.flip_left_right(xy), lambda: xy)

            do_flip_ud = tf.random.uniform([]) < 0.5
            xy = tf.cond(do_flip_ud, lambda: tf.image.flip_up_down(xy), lambda: xy)

            do_rot = tf.random.uniform([]) < 0.25
            xy = tf.cond(do_rot, lambda: tf.image.rot90(xy, k=1), lambda: xy)

            x = xy[..., :DEPTH]
            y = xy[..., DEPTH:]

            # 3. Intensitätsskalierung (0.6 bis 1.0)
            scale = tf.random.uniform([], 0.6, 1.0)
            return x * scale, y * scale
        else:
            # Val/Test sind durch das Caching bereits exakt 256x256
            return x, y

    ds = tf.data.Dataset.from_tensor_slices(valid_starts)
    if is_training: ds = ds.shuffle(len(valid_starts), seed=MY_SEED)

    ds = ds.map(tf_load_window, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(prepare_25d, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augmentieren VOR dem Batching für maximale Diversität!
    ds = ds.map(augment_and_scale, num_parallel_calls=tf.data.AUTOTUNE)

    if not is_training: ds = ds.cache()
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# 7. MAIN LOOP
# =====================================================
def main():
    print(f"\n{'='*60}")
    print(f"🚀 STARTE JOB {args.task_id}/3 | Material: {CURRENT_MATERIAL} | Seed: {MY_SEED}")
    print(f"{'='*60}\n")

    paths = prepare_cache()
    print_peak_ram("Nach Caching")

    train_ds = create_tf_dataset(paths["t_l"], paths["t_h"], True)
    val_ds   = create_tf_dataset(paths["v_l"], paths["v_h"], False)
    test_ds  = create_tf_dataset(paths["test_l"], paths["test_h"], False)

    model = unet_2d_stacked((256, 256, DEPTH))
    optimizer = tf.keras.optimizers.Adam(LR_TARGET, amsgrad=True, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=p14_loss, metrics=[mae_true, mse_true, psnr_true, ssim_true])

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler),
        tf.keras.callbacks.ModelCheckpoint(str(RUN_DIR/f"{RUN_NAME}_best_model.keras"), save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(RUN_DIR/f"{RUN_NAME}_metrics.csv"))
    ]

    print(f"Training startet: {RUN_NAME}")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    print("\n--- TEST EVALUATION (100% UNABHÄNGIG) ---")
    test_results = model.evaluate(test_ds, return_dict=True)
    for k, v in test_results.items(): print(f"  {k}: {v:.4f}")

    meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, LR_TARGET, (256, 256, DEPTH), extra={"material": CURRENT_MATERIAL, "seed": MY_SEED, "test_results": test_results})
    finalize_run(model, history, RUN_NAME, meta, folder_name=str(RUN_DIR))
    print_peak_ram("Final")

if __name__ == "__main__":
    main()