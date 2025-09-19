# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: dl
#     language: python
#     name: python3
# ---

# %%
# ======== Imports =======
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from unet_3d_data import prepare_in_memory_5to5

# %%
# ======== Allocate GPU memory dynamically as needed =======
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

AUTO = tf.data.AUTOTUNE # Chooses optimal number of threads automatically depending on hardware

# %%
# ===== Loading Data in RAM =====
(results, size) = prepare_in_memory_5to5()  # function from 3d_unet_data.py
X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]  # (5, H, W, 1)

# %%
# ======== Making Tensorflow dataset =======

BATCH_SIZE = 16
EPOCHS     = 1

# Sanity check for INPUT_SHAPE
D,H,W,C = INPUT_SHAPE
if (H % 8) or (W % 8):
    print(f"[WARN] H={H} oder W={W} nicht durch 8 teilbar (3x (1,2,2)-Pooling)")

def make_ds(X, Y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

train_ds = make_ds(X_train, Y_train, True)
val_ds   = make_ds(X_val,   Y_val,   False)
test_ds  = make_ds(X_test,  Y_test,  False)


# %%
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

    outputs = layers.Conv3D(1, (1,1,1), activation="sigmoid")(c6)
    return models.Model(inputs, outputs, name="3D_U-Net")



# %%
# =========== Defining Loss function MAE + MS-SSIM (slice-wise) ========

ALPHA = 0.7  # Weight for MS-SSIM

def _flatten_depth(x):
    """
    Making for every depth slice a 2D-image and then evaluate all slices
    (B,D,H,W,C) -> (B*D, H, W, C)
    """
    shape = tf.shape(x)
    b, d = shape[0], shape[1]
    h, w, c = x.shape[2], x.shape[3], x.shape[4]
    return tf.reshape(x, (b*d, h, w, c))

def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    # ohne Ersatz: k <= depth
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0])
    topk = tf.math.top_k(rnd, k=k).indices  # (B,k)
    return topk

def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    """
    Defining MS-SSIM for the loss function equivalently as in the paper
    """
    # y: (B, D, H, W, C)
    b = tf.shape(y_true)[0]
    d = tf.shape(y_true)[1]
    idx = _sample_depth_indices(b, d, k=k)                    # (B,k)
    # sammle ausgewählte Slices
    yt = tf.gather(y_true, idx, batch_dims=1)                 # (B,k,H,W,C)
    yp = tf.gather(y_pred, idx, batch_dims=1)                 # (B,k,H,W,C)
    # zu 2D-Bildern flatten
    yt2 = tf.reshape(yt, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(yp, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    ms  = tf.image.ssim_multiscale(yt2, yp2, max_val=1.0)     # (B*k,)
    return 1.0 - tf.reduce_mean(ms)

def combined_loss(y_true, y_pred, k_slices=1):
    """
    Combining the loss composite of MAE and MS-SSIM
    MAE stable and useful for strong signals --> Bragg peaks
    MS-SSIM focuses on structure --> CDW satellite signals
    """
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss_sampled(y_true, y_pred, k=k_slices)  # k=1 oder 2 ist meist ausreichend
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms

def ms_ssim_metric(y_true, y_pred):
    """
    Showing MS-SSIM metric during training
    """
    yt2 = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape[y_pred)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))


# %%
# ======== Compile model =======

model = unet3d(input_shape=INPUT_SHAPE, base_filters=32)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=combined_loss, metrics=["mae", ms_ssim_metric])
model.summary()


# %%
# ======== Callbacks =======

ckpt_dir = os.path.expanduser("~/data/checkpoints_3d_unet")
os.makedirs(ckpt_dir, exist_ok=True)

cbs = [
    callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best_V2.keras"), monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=2),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=2),
]


# %%
# ======== Train =======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=2
)

