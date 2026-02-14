import os
import sys
import random
import gc
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

# =====================================================
# 0. MIXED PRECISION & PERFORMANCE SETUP
# =====================================================
# BFloat16 nutzt die Tensor-Cores der A100 optimal aus
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)

# Deine Custom-Module (müssen im PYTHONPATH liegen)
from unet_3d_simple_checkpoints import finalize_run, make_meta_dict
from tb_utils import tb_callbacks

# =====================================================
# 1. SETUP & ARGUMENT PARSING
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--point_idx", type=int, required=True, help="Index des Alpha/Beta Paares (0-42)")
args = parser.parse_args()

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

SCRATCH_ROOT = Path.home() / "scratch" / "43_Models_10_Seeds"
TB_ROOT = SCRATCH_ROOT / "tensorboard_logs"
TF_DATA_DIR = Path.home() / "data" / "original_data" / "tfrecords"

for d in [SCRATCH_ROOT, TB_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

def generate_configs():
    configs = []
    for a_idx in range(12): 
        for b_idx in range(13):
            configs.append((round(a_idx / 12, 4), round(b_idx / 12, 4)))
            if len(configs) == 42: break
        if len(configs) == 42: break
    configs.append((1.0, 0.0))
    return configs

ALL_CONFIGS = generate_configs()
MY_POINT_IDX = args.point_idx
MY_ALPHA, MY_BETA = ALL_CONFIGS[MY_POINT_IDX]

SEEDS = range(42, 52) 
BATCH_SIZE = 8
EPOCHS = 200

# =====================================================
# 2. METRIKEN & LOSS (Sicherheits-Casts auf FP32)
# =====================================================
def mae_center(yt, yp):
    yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.abs(yt - yp))

def mse_center(yt, yp):
    yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.square(yt - yp))

def psnr_center(yt, yp):
    yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    mse = tf.reduce_mean(tf.square(yt - yp), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_center(yt, yp):
    yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
    yt, yp = tf.clip_by_value(yt, 0, 1), tf.clip_by_value(yp, 0, 1)
    return tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))

def get_triple_loss(alpha, beta):
    def loss(yt, yp):
        yt, yp = tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        mse = tf.reduce_mean(tf.square(yt - yp))
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(yt, yp, 1.0))
        return (alpha * ssim) + ((1.0 - alpha) * (beta * mse + (1.0 - beta) * mae))
    return loss

# =====================================================
# 3. DATA & ARCHITECTURE
# =====================================================
def parse_tfrecord(example_proto):
    feature_description = {'X': tf.io.FixedLenFeature([], tf.string), 'y': tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    X = tf.reshape(tf.io.decode_raw(parsed['X'], tf.float32), (5, 192, 240, 1))
    y = tf.reshape(tf.io.decode_raw(parsed['y'], tf.float32), (5, 192, 240, 1))
    return X, y

def augment_and_normalize_3d_per_slice(scale_min, scale_max, p=0.5):
    def map_vol(x, y):
        flip = tf.random.uniform([], 0, 1) < p
        x, y = tf.cond(flip, lambda: tf.reverse(x, [2]), lambda: x), tf.cond(flip, lambda: tf.reverse(y, [2]), lambda: y)
        sx, sy = tf.reduce_sum(tf.nn.relu(x), [1,2,3], keepdims=True)+1e-12, tf.reduce_sum(tf.nn.relu(y), [1,2,3], keepdims=True)+1e-12
        sc = tf.random.uniform([], scale_min, scale_max)
        return (x/sx)*sc, (y/sy)*sc
    return map_vol

def prepare_25d_input(x, y):
    return tf.transpose(tf.squeeze(x, -1), [1, 2, 0]), y[tf.shape(y)[0] // 2]

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
    out = layers.Conv2D(1, (1, 1), activation="sigmoid", dtype='float32')(c8) # Wichtig: FP32 am Ende
    return models.Model(inputs, out)

# =====================================================
# 4. MAIN LOOP
# =====================================================
print(f"--- Starte Mixed-Precision TFRecord-Training Punkt {MY_POINT_IDX} ---")

point_dir = SCRATCH_ROOT / f"Point_{MY_POINT_IDX}_a{MY_ALPHA}_b{MY_BETA}"
point_dir.mkdir(exist_ok=True)

for current_seed in SEEDS:
    RUN_NAME = f"P{MY_POINT_IDX}_a{MY_ALPHA:.4f}_b{MY_BETA:.4f}_seed{current_seed}"
    if (point_dir / f"{RUN_NAME}.h5").exists() and (point_dir / f"{RUN_NAME}.json").exists(): continue

    os.environ['PYTHONHASHSEED'] = str(current_seed)
    random.seed(current_seed); np.random.seed(current_seed); tf.random.set_seed(current_seed)
    tf.config.experimental.enable_op_determinism()

    train_ds = (tf.data.TFRecordDataset(str(TF_DATA_DIR / "training.tfrecord"))
                .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
                .cache().shuffle(1000, seed=current_seed)
                .map(augment_and_normalize_3d_per_slice(5000, 15000, 0.5), num_parallel_calls=tf.data.AUTOTUNE)
                .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.TFRecordDataset(str(TF_DATA_DIR / "validation.tfrecord"))
              .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
              .cache().map(augment_and_normalize_3d_per_slice(10000, 10001, 0), num_parallel_calls=tf.data.AUTOTUNE)
              .map(prepare_25d_input, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    model = unet_2d_stacked()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)
    status = {"aborted": False, "reason": "none", "best_psnr": -1.0, "drop_cnt": 0}
    temp_csv = f"temp_{RUN_NAME}.csv"
    best_model_path = point_dir / f"{RUN_NAME}.h5"

    def check_crash(epoch, logs):
        psnr = logs.get('val_psnr_center', 0)
        if psnr > status["best_psnr"]: status["best_psnr"] = psnr; status["drop_cnt"] = 0
        elif epoch >= 10:
            if psnr < (status["best_psnr"] - 4.5) or psnr < 24.0: status["drop_cnt"] += 1
            else: status["drop_cnt"] = 0
            if status["drop_cnt"] >= 3: status["aborted"] = True; status["reason"] = "perf_collapse"; model.stop_training = True

    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor='val_loss', save_best_only=True, verbose=1),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, verbose=1),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                 tf.keras.callbacks.LambdaCallback(on_epoch_end=check_crash),
                 tf.keras.callbacks.CSVLogger(temp_csv), *tb_callbacks(TB_ROOT / RUN_NAME)]

    model.compile(optimizer=optimizer, loss=get_triple_loss(MY_ALPHA, MY_BETA), 
                  metrics=["mae", "mse", mae_center, mse_center, ssim_center, psnr_center])
    
    history = None
    try:
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        if history is not None:
            f_psnr = float(history.history['val_psnr_center'][-1])
            meta = make_meta_dict(RUN_NAME, BATCH_SIZE, EPOCHS, optimizer, 5e-4, (192, 240, 5), 
                                  extra={"alpha": MY_ALPHA, "beta": MY_BETA, "seed": current_seed, 
                                         "aborted": status["aborted"], "reason": status["reason"], "final_psnr": f_psnr})
            finalize_run(model, history, RUN_NAME, meta, folder_name=str(point_dir))
    except Exception as e:
        print(f"Abbruch Seed {current_seed}: {e}")

    if os.path.exists(temp_csv): shutil.move(temp_csv, point_dir / f"{RUN_NAME}_metrics.csv")
    tf.keras.backend.clear_session(); gc.collect()