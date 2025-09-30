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
# unet_3d.py
# ==============================
# 0) Imports & global setup
# ==============================
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)

from tensorflow.keras import regularizers, constraints, layers, models
from unet_3d_data import prepare_in_memory
from train_utils import build_standard_callbacks, clip01

# VRAM dynamisch
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# ==============================
# 1) Daten laden (CPU)
# ==============================
print(">>> Phase 1: Starting data prep on CPU...")
(results, size) = prepare_in_memory(
    data_dir=Path.home() / "data" / "original_data",
    use_vst=False, size=5, group_len=41, dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]  # (D,H,W,C)
BATCH_SIZE = 32
EPOCHS     = 200

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

# ==============================
# 2) Model (3D U-Net)
# ==============================
def conv_block(x, filters, kernel_size=(3,3,3), padding="same"):
    ki  = "he_normal"; kr = regularizers.l2(1e-5); kc = constraints.MaxNorm(3.0)
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x); x = layers.ELU()(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x); x = layers.ELU()(x)
    return x

def unet3d(input_shape=(5,192,240,1), base_filters=8):
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, base_filters);   p1 = layers.MaxPooling3D((1,2,2))(c1)
    c2 = conv_block(p1, base_filters*2);     p2 = layers.MaxPooling3D((1,2,2))(c2)
    c3 = conv_block(p2, base_filters*4);     p3 = layers.MaxPooling3D((1,2,2))(c3)
    bn = conv_block(p3, base_filters*8)
    u3 = layers.Conv3DTranspose(base_filters*4, (1,2,2), (1,2,2), padding="same")(bn)
    u3 = layers.Concatenate()([u3, c3]);     c4 = conv_block(u3, base_filters*4)
    u2 = layers.Conv3DTranspose(base_filters*2, (1,2,2), (1,2,2), padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2]);     c5 = conv_block(u2, base_filters*2)
    u1 = layers.Conv3DTranspose(base_filters,   (1,2,2), (1,2,2), padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1]);     c6 = conv_block(u1, base_filters)
    out = layers.Conv3D(1, (1,1,1), activation="sigmoid",
                        kernel_initializer="glorot_uniform")(c6)
    return models.Model(inputs, out, name="3D_U-Net-ELU-LN")

# ==============================
# 3) Loss & Metrics (MAE + MS-SSIM sampled)
# ==============================
ALPHA = 0.7  # Gewicht für MS-SSIM

def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0])
    return tf.math.top_k(rnd, k=k).indices  # (B,k)

def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    b = tf.shape(y_true)[0]; d = tf.shape(y_true)[1]
    idx = _sample_depth_indices(b, d, k=k)        # (B,k)
    yt = tf.gather(y_true, idx, batch_dims=1)     # (B,k,H,W,C)
    yp = tf.gather(y_pred, idx, batch_dims=1)
    H = tf.shape(y_true)[2]; W = tf.shape(y_true)[3]; C = tf.shape(y_true)[4]
    yt4 = clip01(tf.reshape(yt, (-1, H, W, C)))
    yp4 = clip01(tf.reshape(yp, (-1, H, W, C)))
    ms  = tf.image.ssim_multiscale(yt4, yp4, max_val=1.0)  # (B*k,)
    return 1.0 - tf.reduce_mean(ms)

def combined_loss(y_true, y_pred, k_slices=1):
    yt = clip01(y_true); yp = clip01(y_pred)
    l_mae = tf.reduce_mean(tf.abs(yt - yp))
    l_ms  = ms_ssim_loss_sampled(yt, yp, k=k_slices)
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms

def ms_ssim_metric(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    yt2 = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp2 = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))

def psnr_metric(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)

# ==============================
# 4) Naming pipeline (einheitlich)
# ==============================




# ==============================
# 5) Callbacks (gemeinsam)
# ==============================


# ==============================
# 6) Compile + Training
# ==============================
model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=["mae","mse", psnr_metric, ms_ssim_metric],
    jit_compile=False
)

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 20},
    "data_prep": {"use_vst": False, "size": 5, "group_len": 41, "dtype": "float32"},
    "alpha": 0.7,                                   # Gewichte für kombinierten Loss
    "loss_components": {"mae": 0.3, "ms_ssim": 0.7} # explizit MS-SSIM
}


cbs, bf, ckpt_best = build_standard_callbacks(
    ckpt_root=ckpt_root,
    run_meta=run_meta,
    monitor="val_loss",
    patience_es=20,
    reduce_on_plateau=True,
    reduce_factor=0.5,
    reduce_patience=10,
    min_lr=1e-6,
    include_nan_guards=True,
    include_logger=True,
    code_name="AUTO",
    verbose_ckpt=1
)

print(">>> Phase 3: GPU training starts now!")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=2)
print(">>> Training complete")

final_val = model.evaluate(val_ds, return_dict=True, verbose=0)
print("FINAL VAL:", {k: float(v) for k, v in final_val.items()})
final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})

