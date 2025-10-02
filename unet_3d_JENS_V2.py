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
# unet_3d_JENS.py
# ==============================
# 0) Imports & global setup
# ==============================
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)  # XLA aus

from tensorflow.keras import regularizers, constraints, layers, models, callbacks
from tensorflow.keras.optimizers import AdamW

from unet_3d_data_JENS import prepare_in_memory_5to5
from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_standard_callbacks, clip01

# Reproduzierbarkeit
seed = 0
reset_random_seeds(seed)

# VRAM dynamisch
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# ==============================
# 1) Daten laden (CPU)
# ==============================
print(">>> Phase 1: Starting data prep on CPU...")
results = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    size=5, group_len=41, dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]   # (D,H,W,C)
BATCH_SIZE = 32
EPOCHS     = 200

# ==============================
# 2) Preprocessing & Augmentation
# ==============================
preproc_train_slice = SumScaleNormalizer(
    scale_range=[5000, 15001], pre_offset=0.0, normalize_label=True,
    axis=(1, 2, 3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0., 1.]
)
preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001], pre_offset=0.0, normalize_label=True,
    axis=(1, 2, 3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0., 1.]
)

def map_slice_wise(normalizer):
    def _finite01(t):
        t = tf.cast(t, tf.float32)
        t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
        return tf.clip_by_value(t, 0.0, 1.0)
    def _fn(x, y):
        x_norm, y_norm = normalizer.map(x, y)
        return _finite01(x_norm), _finite01(y_norm)
    return _fn

def augment_5stack_flips(x, y):
    do_lr = tf.random.uniform(()) < 0.5
    do_ud = tf.random.uniform(()) < 0.5
    def fliplr(t): return tf.reverse(t, axis=[2])  # W
    def flipud(t): return tf.reverse(t, axis=[1])  # H
    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    x = tf.cond(do_ud, lambda: flipud(x), lambda: x)
    y = tf.cond(do_ud, lambda: flipud(y), lambda: y)
    return x, y

# ==============================
# 3) Datasets (mit NaN-Guard)
# ==============================
def nan_debug(x, y):
    nx = tf.reduce_sum(tf.cast(~tf.math.is_finite(x), tf.int32))
    ny = tf.reduce_sum(tf.cast(~tf.math.is_finite(y), tf.int32))
    tf.debugging.assert_equal(nx, 0, message="NaN/Inf in X batch")
    tf.debugging.assert_equal(ny, 0, message="NaN/Inf in Y batch")
    return x, y

def make_ds(X, Y, *, shuffle=True, preproc=None, augmenter=None,
            limit=None, cache_in_memory=False, check_nans=False,
            shuffle_buf=512, prefetch_n=2, num_calls=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if preproc is not None:
        ds = ds.map(lambda x, y: tuple(preproc(x, y)), num_parallel_calls=num_calls)
    if cache_in_memory:
        ds = ds.cache()
    if augmenter is not None:
        ds = ds.map(lambda x, y: augmenter(x, y), num_parallel_calls=num_calls)
    if check_nans:
        ds = ds.map(nan_debug, num_parallel_calls=num_calls)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(shuffle_buf, X.shape[0]), reshuffle_each_iteration=True)
    if limit is not None:
        ds = ds.take(int(limit))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_n)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, shuffle=True,
                   preproc=map_slice_wise(preproc_train_slice),
                   augmenter=augment_5stack_flips, check_nans=True)
val_ds   = make_ds(X_val, Y_val, shuffle=False,
                   preproc=map_slice_wise(preproc_valid_slice),
                   augmenter=None, check_nans=True)
test_ds  = make_ds(X_test, Y_test, shuffle=False,
                   preproc=map_slice_wise(preproc_valid_slice),
                   augmenter=None, check_nans=True)
print(">>> Datasets created")

# ==============================
# 4) Model (3D U-Net)
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
# 5) Loss & Metrics  (MAE + SSIM)
# ==============================

def _to_4d(yt, yp):
    b,d,h,w,c = tf.unstack(tf.shape(yt))
    return tf.reshape(yt, (b*d, h, w, c)), tf.reshape(yp, (b*d, h, w, c))

def ssim_3d_mean_safe(y_true, y_pred, max_val=1.0,
                      filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    yt = clip01(y_true); yp = clip01(y_pred)
    yt4, yp4 = _to_4d(yt, yp)
    v = tf.image.ssim(yp4, yt4, max_val=max_val,
                      filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
    return tf.reduce_mean(v)

def ssim_metric(y_true, y_pred): return ssim_3d_mean_safe(y_true, y_pred, max_val=1.0)
ssim_metric.__name__ = "ssim"

def psnr_metric(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)
psnr_metric.__name__ = "psnr"

class CombinedMAE_SSIM_Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, name="mae_ssim"):
        super().__init__(name=name); self.alpha = float(alpha)
    def call(self, y_true, y_pred):
        yt = clip01(y_true); yp = clip01(y_pred)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        ssimv = ssim_3d_mean_safe(yt, yp, max_val=1.0)
        return (1.0 - self.alpha) * mae + self.alpha * (1.0 - ssimv)

# ==============================
# 6) Compile
# ==============================
model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=AdamW(learning_rate=1e-4),
    loss=CombinedMAE_SSIM_Loss(alpha=0.7),
    metrics=[ssim_metric, tf.keras.metrics.MeanAbsoluteError(name="mae"),
             tf.keras.metrics.MeanSquaredError(name="mse"), psnr_metric],
    jit_compile=False
)

# ==============================
# 7) Naming pipeline (einheitlich)
# ==============================




# ==============================
# 8) Callbacks (gemeinsam)
# ==============================
ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 20},
    "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
    "alpha": 0.7,                             # Gewichte für kombinierten Loss
    "loss_components": {"mae": 0.3, "ssim": 0.7}  # einfache SSIM
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
    code_name="unet_3d_JENS",
    verbose_ckpt=1
)

# ==============================
# 9) Train
# ==============================
print(">>> Phase 3: GPU training starts now!")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=0)
print(">>> Phase 3: Training complete!")

# ==============================
# 10) Evaluate
# ==============================
final_val = model.evaluate(val_ds, return_dict=True, verbose=0)
print("FINAL VAL:", {k: float(v) for k, v in final_val.items()})
final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})

