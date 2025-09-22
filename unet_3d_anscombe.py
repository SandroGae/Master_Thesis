# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
# ======== Imports =======
import os

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") # Increases performance without loss of quality (calculations still done with float_32 precision)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from unet_3d_data import prepare_in_memory_5to5
from pathlib import Path
# ======== Allocate GPU memory dynamically as needed =======
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

AUTO = tf.data.AUTOTUNE # Chooses optimal number of threads automatically depending on hardware
# ===== Loading Data in RAM =====

print(">>> Phase 1: Starting data prep on CPU...")
(results, size) = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    use_vst=True, # With anscombe transform
    size=5,
    group_len=41,
    dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]


INPUT_SHAPE = X_train.shape[1:]  # (5, H, W, 1)
# ======== Making Tensorflow dataset =======

BATCH_SIZE = 16
EPOCHS     = 400

# Sanity check for INPUT_SHAPE
D,H,W,C = INPUT_SHAPE
if (H % 8) or (W % 8):
    print(f"[WARN] H={H} oder W={W} nicht durch 8 teilbar (3x (1,2,2)-Pooling)")

def make_ds(X, Y, shuffle=True):
    """
    Creates a tensorflow dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, True)
val_ds   = make_ds(X_val,   Y_val,   False)
test_ds  = make_ds(X_test,  Y_test,  False)
print(">>> Datasets created")
# ========= Defining 3D-U-Net Architecture ========

def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def unet3d(input_shape=(5, 192, 240, 1), base_filters=32):
    inputs = layers.Input(shape=input_shape)

    # Encoder (pool only over H,W)
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c3)

    # Bottleneck
    bn = conv_block(p3, base_filters*8)

    # Decoder (upsample only over H,W)
    u3 = layers.Conv3DTranspose(base_filters*4, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(bn) # bottleneck
    u3 = layers.concatenate([u3, c3])
    c4 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = conv_block(u1, base_filters)

    outputs = layers.Conv3D(1, (1,1,1), dtype="float32", activation=None)(c6)  # linear activation
    return models.Model(inputs, outputs, name="3D_U-Net")

# =========== Defining Metrics for training ========

def inv_anscombe(z):
    z = tf.maximum(z, 1e-3)
    z2 = z / 2.0
    # Unbiased approx. (gekürzt) fuer Poisson
    y = z2**2 - 1.0/8.0 + 1.0/(4.0*z**2) - 11.0/(8.0*z**4)
    return tf.nn.relu(y)

def psnr_orig_metric(y_true_vst, y_pred_vst):
    y_true = tf.clip_by_value(inv_anscombe(y_true_vst), 0.0, 1.0)
    y_pred = tf.clip_by_value(inv_anscombe(y_pred_vst), 0.0, 1.0)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ms_ssim_orig_metric(y_true_vst, y_pred_vst):
    yt = inv_anscombe(y_true_vst)
    yp = inv_anscombe(y_pred_vst)
    yt2 = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp2 = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))



# ======== Callbacks =======

ckpt_dir = os.path.expanduser("~/data/checkpoints_3d_unet")
os.makedirs(ckpt_dir, exist_ok=True)

cbs = [
    callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best_V2.keras"), monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
]
# ======== Train =======

print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mae", "mse", psnr_orig_metric, ms_ssim_orig_metric],
    jit_compile=False # Would be false per default, but just to be sure
)
# model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=2
)
print(">>> Training complete")

