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
"""
Durchsuchter Raum:
DEPTHS      = [2, 3, 4]                         Architektur-Tiefe
BASE_FILTER = [8, 16, 24]                       Start-Kanaele
OUT_ACTS    = ["sigmoid", "tanh", "linear"]     Output-Aktivierung
MAX_EPOCHS_SCOUT = 10

Bestes Ergebnis:
depth 3, base_filters 16, output_activation tanh
"""

# %%
# unet_3d_JENS.py
# ==============================
# 0) Imports & global setup
# ==============================

from pathlib import Path
import numpy as np
import tensorflow as tf
from pathlib import Path
tf.config.optimizer.set_jit(False)  # XLA aus

import re
from pathlib import Path
from tensorflow.keras.callbacks import CSVLogger, Callback
import h5py, numpy as np, tensorflow as tf

from tensorflow.keras import regularizers, constraints, layers, models
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

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

# %%
# ==============================
# 1–3) Daten-Streaming (kein RAM-Fullload)
# ==============================
import h5py

class H5Dataset:
    """Lazy Loader, liest 2D-Frames oder 5-Stacks on-demand aus HDF5."""
    def __init__(self, path: Path, stack_depth=5, dtype=np.float16):
        self.f = h5py.File(path, "r")
        self.high = self.f["/high_count/data"]  # (H,W,N)
        self.low  = self.f["/low_count/data"]
        self.H, self.W, self.N = self.high.shape
        self.D = 1  # aktuell 1 Slice, falls du echte 5-Stacks willst: D=5
        self.dtype = dtype

    def __len__(self): return self.N

    def get_pair(self, i):
        hi = self.high[..., i].astype(self.dtype)
        lo = self.low[...,  i].astype(self.dtype)
        hi = hi[None, ..., None]  # (1,H,W,1)
        lo = lo[None, ..., None]
        return lo, hi


def make_stream_ds(h5obj, *, batch_size=32, shuffle=False,
                   preproc=None, augmenter=None, prefetch=1):
    """tf.data.Dataset-Wrapper um H5Dataset."""
    def gen():
        for i in range(len(h5obj)):
            x, y = h5obj.get_pair(i)
            yield x, y

    out_spec = (
        tf.TensorSpec(shape=(1, h5obj.H, h5obj.W, 1), dtype=tf.float16),
        tf.TensorSpec(shape=(1, h5obj.H, h5obj.W, 1), dtype=tf.float16),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_spec)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                num_parallel_calls=1)

    if preproc is not None:
        ds = ds.map(preproc, num_parallel_calls=1)
    if augmenter is not None:
        ds = ds.map(augmenter, num_parallel_calls=1)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(512, len(h5obj)),
                        reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(prefetch)


print(">>> Phase 1: Opening HDF5 datasets (streaming mode)...")

DATA_DIR = Path.home() / "data" / "original_data"
train_h5 = H5Dataset(DATA_DIR / "training_data.hdf5", stack_depth=5, dtype=np.float16)
val_h5   = H5Dataset(DATA_DIR / "validation_data.hdf5", stack_depth=5, dtype=np.float16)
test_h5  = H5Dataset(DATA_DIR / "test_data.hdf5",       stack_depth=5, dtype=np.float16)

INPUT_SHAPE = (train_h5.D, train_h5.H, train_h5.W, 1)
BATCH_SIZE  = 32
EPOCHS      = 0

# ==============================
# Preprocessing & Augmentation (wie vorher)
# ==============================

# Normalisierung
preproc_train_slice = SumScaleNormalizer(
    scale_range=[5000, 15001],  # Training: zufaellig in diesem Bereich
    pre_offset=0.0,
    normalize_label=True,
    axis=(1, 2, 3),
    batch_mode=True,
    clip_before=[0., float("inf")],
    clip_after=[0., 1.]
)

preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001],   # Val/Test: fixe Skalierung
    pre_offset=0.0,
    normalize_label=True,
    axis=(1, 2, 3),
    batch_mode=True,
    clip_before=[0., float("inf")],
    clip_after=[0., 1.]
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
    # flip entlang H und W (Achsen 1/2 bei Tensorform (D,H,W,C))
    do_lr = tf.random.uniform(()) < 0.5  # left-right (W)
    do_ud = tf.random.uniform(()) < 0.5  # up-down (H)

    def fliplr(t): return tf.reverse(t, axis=[2])  # W
    def flipud(t): return tf.reverse(t, axis=[1])  # H

    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    x = tf.cond(do_ud, lambda: flipud(x), lambda: x)
    y = tf.cond(do_ud, lambda: flipud(y), lambda: y)
    return x, y

print(">>> Phase 2: Create Tensorflow Datasets (streaming)...")

train_ds = make_stream_ds(
    train_h5, batch_size=BATCH_SIZE, shuffle=True,
    preproc=map_slice_wise(preproc_train_slice),
    augmenter=augment_5stack_flips, prefetch=1
)
val_ds = make_stream_ds(
    val_h5, batch_size=BATCH_SIZE, shuffle=False,
    preproc=map_slice_wise(preproc_valid_slice), prefetch=1
)
test_ds = make_stream_ds(
    test_h5, batch_size=BATCH_SIZE, shuffle=False,
    preproc=map_slice_wise(preproc_valid_slice), prefetch=1
)

print(">>> Datasets created (streaming)")


# %%
# ==============================
# 4) Model Architektur (Depth=4)
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

def unet3d(input_shape=(5,192,240,1), base_filters=16, output_activation="tanh"):
    inputs = layers.Input(shape=input_shape)

    # Encoder (Depth=4)
    c1 = conv_block(inputs, base_filters)            ; p1 = layers.MaxPooling3D((1,2,2))(c1)
    c2 = conv_block(p1, base_filters*2)              ; p2 = layers.MaxPooling3D((1,2,2))(c2)
    c3 = conv_block(p2, base_filters*4)              ; p3 = layers.MaxPooling3D((1,2,2))(c3)
    c4 = conv_block(p3, base_filters*8)              ; p4 = layers.MaxPooling3D((1,2,2))(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters*16)

    # Decoder
    u4 = layers.Conv3DTranspose(base_filters*8, (1,2,2), (1,2,2), padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])              ; c5 = conv_block(u4, base_filters*8)

    u3 = layers.Conv3DTranspose(base_filters*4, (1,2,2), (1,2,2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])              ; c6 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, (1,2,2), (1,2,2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])              ; c7 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters,   (1,2,2), (1,2,2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])              ; c8 = conv_block(u1, base_filters)

    out = layers.Conv3D(1, (1,1,1), activation=output_activation,
                        kernel_initializer="glorot_uniform")(c8)

    return models.Model(inputs, out,
    name=f"3D_U-Net_JENS_d4_bf{base_filters}_out{output_activation}"
)


# %%
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



# %%
# ==============================
# 6) Compile
# ==============================

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16, output_activation="tanh")
model.compile(optimizer=AdamW(learning_rate=1e-4),
    loss=CombinedMAE_SSIM_Loss(alpha=0.7),
    metrics=[ssim_metric, tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse"), psnr_metric],
    jit_compile=False
)

# %%
# ==============================
# 7) Callbacks + CSV_logging
# ==============================

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 200},
    "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
    "alpha": 0.7,                             # Gewichte für kombinierten Loss
    "loss_components": {"mae": 0.3, "ssim": 0.7}  # einfache SSIM
}

cbs, bf, ckpt_best = build_standard_callbacks(
    ckpt_root=ckpt_root,
    run_meta=run_meta,
    monitor="val_loss",
    patience_es=200,
    reduce_on_plateau=True,
    reduce_factor=0.5,
    reduce_patience=10,
    min_lr=1e-6,
    include_nan_guards=True,
    include_logger=True,
    code_name="unet_3d_JENS",
    verbose_ckpt=1
)

# CSV logger
CSV_DIR = Path.home() / "data" / "logs_csv"

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = CSV_DIR / f"{bf.code}_train_{stamp}.csv"

csv_cb = CSVLogger(filename=str(csv_path), separator=",", append=False)
# CSV-Logger zu den Callbacks packen
cbs = list(cbs) + [csv_cb]

# %%
# ==============================
# 8) Training + kurze Evaluierung
# ==============================
print(">>> Phase 3: GPU training starts now!")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=0)
print(">>> Phase 3: Training complete!")

final_val = model.evaluate(val_ds, return_dict=True, verbose=0)
print("FINAL VAL:", {k: float(v) for k, v in final_val.items()})
final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})


# %%
# ==============================
# 11) Mini-Sweep: Tiefe, Base-Filters, Output-Activation, Loss (hier fix)
#     Pro Run 10 Epochen, JSON-Report + CSV Logger
# ==============================
CSV_DIR_SWEEP = Path.home() / "data" / "logs_csv_scout"
CSV_DIR_SWEEP.mkdir(parents=True, exist_ok=True)

def _safe_tag(s: str) -> str:
    # Erlaubt nur A-Za-z0-9_.\\/>- ; ersetze andere Zeichen durch '_'.
    # Garantiert, dass der erste Char gueltig ist (A-Za-z oder Ziffer oder '.').
    # Ersetze unerlaubte Zeichen
    t = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", s)
    # falls erster Char ungueltig, prefix mit 'A'
    if not re.match(r"^[A-Za-z0-9.]", t):
        t = "A" + t
    return t

# --- (A) kleine Hilfen: norm + act ohne dein conv_block umzuschreiben ---
def conv_block_param(x, filters, *, norm="LN", act="ELU"):
    if norm.upper() == "LN" and act.upper() == "ELU":
        return conv_block(x, filters)  # exakt dein Block
    ki  = "he_normal"; kr = regularizers.l2(1e-5); kc = constraints.MaxNorm(3.0)
    y = layers.Conv3D(filters, (3,3,3), padding="same",
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    y = (layers.LayerNormalization(epsilon=1e-5)(y) if norm.upper()=="LN"
         else layers.BatchNormalization()(y))
    y = (layers.ELU()(y) if act.upper()=="ELU" else layers.LeakyReLU(alpha=0.1)(y))
    y = layers.Conv3D(filters, (3,3,3), padding="same",
                      kernel_initializer=ki, use_bias=False,
                      kernel_regularizer=kr, kernel_constraint=kc)(y)
    y = (layers.LayerNormalization(epsilon=1e-5)(y) if norm.upper()=="LN"
         else layers.BatchNormalization()(y))
    y = (layers.ELU()(y) if act.upper()=="ELU" else layers.LeakyReLU(alpha=0.1)(y))
    return y

# --- (B) U-Net mit variabler Tiefe (depth_levels = Anzahl Downsamplings) ---
def build_unet3d_depth(input_shape,
                       *, base_filters=16, depth_levels=3,
                       norm="LN", act="ELU",
                       output_activation="sigmoid",
                       name=None):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    skips = []
    x = inputs
    filters = base_filters
    for _ in range(depth_levels):
        x = conv_block_param(x, filters, norm=norm, act=act)
        skips.append(x)
        x = layers.MaxPooling3D((1,2,2))(x)
        filters *= 2
    # Bottleneck
    x = conv_block_param(x, filters, norm=norm, act=act)
    # Decoder
    for level in reversed(range(depth_levels)):
        filters //= 2
        x = layers.Conv3DTranspose(filters, (1,2,2), (1,2,2), padding="same")(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block_param(x, filters, norm=norm, act=act)
    out = layers.Conv3D(1, (1,1,1),
                        activation=output_activation,
                        kernel_initializer="glorot_uniform")(x)
    model_name = name or f"UNet3D_d{depth_levels}_bf{base_filters}_{act}_{norm}_out{output_activation}"
    model_name = _safe_tag(model_name)
    return models.Model(inputs, out, name=model_name)

# --- (C) Loss fix: MAE+SSIM(0.7) ---
def get_loss_fixed():
    return CombinedMAE_SSIM_Loss(alpha=0.7, name="mae_ssim_0p7")

# --- (D) Suchraum ---
DEPTHS       = [3, 4]                 # Architektur-Tiefe
BASE_FILTERS = [8, 16, 24]            # Start-Kanaele
OUT_ACTS     = ["sigmoid"]            # Targets ∈ [0,1] -> sigmoid
LEARNING_RATES = [3e-4, 1e-4, 3e-5]   # kleiner LR-Sweep lohnt mehr als bf=32
BATCH_SIZES = [32, 128]               # Batch-Grössen

MAX_EPOCHS_SCOUT = 2

class InjectStatic(Callback):
    def __init__(self, **static): super().__init__(); self.static = static
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None: logs.update(self.static)

def run_mini_sweep():
    runs = 0
    ckpt_root_scout = Path.home() / "data" / "checkpoints_3d_unet_scout"
    CSV_DIR_SWEEP = Path.home() / "data" / "logs_csv_scout"; CSV_DIR_SWEEP.mkdir(parents=True, exist_ok=True)

    for depth in DEPTHS:
      for bf in BASE_FILTERS:
        for outa in OUT_ACTS:
          for lr in LEARNING_RATES:
            for bs in BATCH_SIZES:
              runs += 1
              raw_tag = f"d{depth}_bf{bf}_ELU_LN_out{outa}_lr{lr:g}_bs{bs}__mae+ssim:0.7"
              tag = _safe_tag(raw_tag)

              # Datasets fuer diese Batchgroesse
              train_ds_bs = make_stream_ds(train_h5, batch_size=bs, shuffle=True,
                                        preproc=map_slice_wise(preproc_train_slice),
                                        augmenter=augment_5stack_flips, prefetch=1)
              val_ds_bs = make_stream_ds(val_h5, batch_size=bs, shuffle=False,
                                        preproc=map_slice_wise(preproc_valid_slice), prefetch=1)
              test_ds_bs = make_stream_ds(test_h5, batch_size=bs, shuffle=False,
                                        preproc=map_slice_wise(preproc_valid_slice), prefetch=1)


              # Modell
              model = build_unet3d_depth(
                  INPUT_SHAPE, base_filters=bf, depth_levels=depth,
                  norm="LN", act="ELU", output_activation=outa, name=f"Scout_{tag}"
              )
              model.compile(
                  optimizer=AdamW(learning_rate=lr),
                  loss=get_loss_fixed(),
                  metrics=[ssim_metric,
                           tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.MeanSquaredError(name="mse"),
                           psnr_metric],
                  jit_compile=False
              )

              # Callbacks (ES aus, ReduceLROnPlateau aus)
              run_meta_scout = {
                  "batch_size": bs,
                  "epochs": MAX_EPOCHS_SCOUT,
                  "early_stopping": {"monitor": "val_loss", "patience": 10**9},
                  "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
                  "alpha": 0.7, "loss_components": {"mae": 0.3, "ssim": 0.7}
              }
              cbs_scout, _, _ = build_standard_callbacks(
                  ckpt_root=ckpt_root_scout,
                  run_meta=run_meta_scout,
                  monitor="val_loss",
                  patience_es=10**9,
                  reduce_on_plateau=False,
                  include_nan_guards=True,
                  include_logger=True,
                  code_name=f"sweep_{tag}",
                  verbose_ckpt=0
              )

              # CSV + statische Spalten
              stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # falls du Option A unten nutzt
              csv_path = CSV_DIR_SWEEP / f"sweep_{tag}_{stamp}.csv"
              csv_cb = CSVLogger(str(csv_path), separator=",", append=False)
              inject = InjectStatic(run_tag=tag, depth=depth, base_filters=bf,
                                    out_act=outa, lr=lr, batch_size=bs)

              cbs_scout = [inject] + list(cbs_scout) + [csv_cb]

              print(f"\n[SWEEP] Run {runs}: {raw_tag}")
              history = model.fit(train_ds_bs, validation_data=val_ds_bs,
                    epochs=MAX_EPOCHS_SCOUT, callbacks=cbs_scout, verbose=0,
                    workers=1, use_multiprocessing=False, max_queue_size=4)
              tf.keras.backend.clear_session(); import gc; gc.collect()

              _ = model.evaluate(val_ds_bs,  return_dict=True, verbose=0)
              _ = model.evaluate(test_ds_bs, return_dict=True, verbose=0)

              tf.keras.backend.clear_session()


    print(f"\n[SWEEP] Completed {runs} runs. CSVs in {CSV_DIR_SWEEP}")

# Start
run_mini_sweep()
