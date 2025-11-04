# %%
# zzz_unet_3d.py  MAE only

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras import regularizers, constraints, layers, models, optimizers

from unet_3d_data import prepare_in_memory
from train_utils import (build_standard_callbacks, clip01)


# Lädt Daten in GPU dynamisch nach Bedarf
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# %%
# ===== Daten laden =====

print(">>> Phase 1: Starting data prep on CPU...")
(results, prep_meta) = prepare_in_memory(
    data_dir=Path.home() / "data" / "original_data",
    size=5,
    group_len=41,
    percentile=99.9,
    dtype=np.float32,
)
print(">>> Data preparation finished, all data in RAM")

# %%
# ===== Datenset erstellen =====

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]  # (5,H,W,1)
BATCH_SIZE  = 32
EPOCHS      = 200

D,H,W,C = INPUT_SHAPE
if (H % 8) or (W % 8):
    print(f"[WARN] H={H} oder W={W} nicht durch 8 teilbar (3x (1,2,2)-Pooling)")

def make_ds(X, Y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle: ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, True)
val_ds   = make_ds(X_val,   Y_val,   False)
test_ds  = make_ds(X_test,  Y_test,  False)
print(">>> Datasets created")


# %%
# ===== Modell Architektur =====

def conv_block(x, filters, kernel_size=(3,3,3), padding="same"):
    ki  = "he_normal"
    kr  = regularizers.l2(1e-5)
    kc  = constraints.MaxNorm(3.0)

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.ELU()(x)

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.ELU()(x)
    return x

def unet3d(input_shape=(5, 192, 240, 1), base_filters=16):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c3)

    bn = conv_block(p3, base_filters*8) # bottleneck

    u3 = layers.Conv3DTranspose(base_filters*4, (1,2,2), (1,2,2), padding="same")(bn)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, (1,2,2), (1,2,2), padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters, (1,2,2), (1,2,2), padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, base_filters)

    outputs = layers.Conv3D(1, (1,1,1), activation="sigmoid",
                            kernel_initializer="glorot_uniform")(c6)
    return models.Model(inputs, outputs, name="3D_U-Net_ELU_LN_sigmoid")


# %%
# ===== Metrik zur Evaluierung =====

def psnr_metric(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)


# %%
# ===== Training =====

print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="mae",
    metrics=["mae", "mse", psnr_metric],
)

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"

# Vorbereiten für JSON
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 20},
    "data_prep": prep_meta,
    "alpha": None,
    "loss_components": ["mae"],
}

callbacks_list, bf, ckpt_best = build_standard_callbacks(
    ckpt_root=ckpt_root,
    run_meta=run_meta,
    monitor="val_loss",
    patience_es=20,
    reduce_on_plateau=True,
    reduce_factor=0.5,
    reduce_patience=5,
    min_lr=1e-6,
    include_nan_guards=True,
    include_logger=True,
    code_name="unet_3d",
    verbose_ckpt=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=2
)

print(">>> Training complete")

