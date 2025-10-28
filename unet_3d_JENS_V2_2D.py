# unet_3d_JENS_V2_2D.py
# ==============================
# 0) Imports & global setup
# ==============================
import os
# Deaktiviere XLA (just in time compiler) aus Stabilitätsgründen
os.environ["TF_DISABLE_XLA"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
# Behebe "Failed to allocate scratch space errors (testet verschiedene Faltungsalgorithmen bei der ersten Iteration)"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# Etfernt hartes Workspace-Limit (512 MB war zu klein)
os.environ.pop("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", None)
# GPU alloziert nur benötigte Menge an VRAM
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Weniger TF/C++-Spam (INFO+WARNING weg, ERROR bleibt)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
# Unterdrückt nervige Warnungen im Log
tf.get_logger().setLevel("ERROR")
from absl import logging as absl_logging
# XLA/absl-errors unterdrücken
absl_logging.set_verbosity(absl_logging.FATAL)

from pathlib import Path
import math
import re
from tensorflow.keras.callbacks import CSVLogger, Callback
from tensorflow.keras import regularizers, constraints, layers, models
from tensorflow.keras.optimizers import AdamW
from datetime import datetime

from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_1stack_datasets_flat, clip01, build_standard_callbacks


# Reproduzierbarkeit
seed = 0
reset_random_seeds(seed)
AUTO = tf.data.AUTOTUNE # tensorflow wählt selbst wie viele elemente parallel geladen/verarbeitet werden

# %%
# ==============================
# Daten-Streaming (kein RAM-Fullload)
# ==============================

# Normalisierung identisch zu Jens (bis auf 10'000 bei val)
preproc_train_slice = SumScaleNormalizer(
    scale_min=5000, scale_max=15000,
    pre_offset=0.0, 
    normalize_label=True, 
    batch_mode=False # 4D input (D,H,W,C) --> samples werden einzeln normalisiert
)
preproc_valid_slice = SumScaleNormalizer(
    scale_min=10000, 
    scale_max=10001,
    pre_offset=0.0, 
    normalize_label=True, 
    batch_mode=False
)


BATCH_SIZE = 32
EPOCHS = 0

def map_slice_wise(normalizer):
    def _finite01(t):
        # Sichert Robustheit gegen NaN/Inf
        t = tf.cast(t, tf.float32) # alle Operationen in float32
        t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)) # NaN/Inf -> 0
        return tf.clip_by_value(t, 0.0, 1.0) # Clip [0,1]
    def _fn(x, y):
        # Slice-weise Normalisierung (pro d über H,W,C), danach Sicherheits-Clip [0,1]
        x_norm, y_norm = normalizer.map(x, y)
        return _finite01(x_norm), _finite01(y_norm)
    return _fn

def augment_fliplr_only(x, y):
    # Augmentation: nur Left-Right wie bei Jens
    do_lr = tf.random.uniform(()) < 0.5
    fliplr = lambda t: tf.reverse(t, axis=[2])  # 4D: (D,H,W,C) -> W ist axis 2 for 5D: (B,D,H,W,C) -> W is axis 3!!!
    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    return x, y



def pipeline_train(x, y):
    # Pipelines: Augment -> Normalize -> Sicherheits-Clip
    x, y = augment_fliplr_only(x, y)                     # 1) augment
    x, y = preproc_train_slice.map(x, y)                 # 2) normalize
    return clip01(x), clip01(y)                          # 3) safety

def pipeline_val(x, y):
    # Jens augmentiert auch Validation
    x, y = augment_fliplr_only(x, y)                     # 1) augment
    x, y = preproc_valid_slice.map(x, y)                 # 2) normalize
    return clip01(x), clip01(y)                          # 3) safety


print(">>> Phase 1: Baue Datensatz (flat, D=1)...")
train_ds, val_ds, test_ds, meta = build_1stack_datasets_flat(
    data_dir=Path.home() / "data" / "original_data",
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    preproc_train=pipeline_train,
    preproc_eval=pipeline_val,
    augmenter=None,
    deterministic=False,
    cache_after_preproc=False
)


INPUT_SHAPE = meta["input_shape"]
print(">>> Datasets created (D =", meta["D"], ")")

def _steps(meta, split, batch):
    N = {"train": meta["n_train"], "val": meta["n_val"], "test": meta["n_test"]}[split]
    import math
    return math.ceil(N / batch)




# %%
# ==============================
# Model Architektur (Depth=4)
# ==============================

def conv_block(x, filters, kernel_size=(3,3,3), padding="same"):
    ki  = "he_normal"; # Gewichtsinitialisierung
    # kr = regularizers.l2(1e-5); # Zusatzterm zur Verlustfunktion "lambda"
    # kc = constraints.MaxNorm(3.0) # Max-Norm Regularisierung der Gewichte auf ||v|| <= 3.0
    x = layers.Conv3D(filters, kernel_size, padding=padding, kernel_initializer=ki, 
                      use_bias=True, activation="relu")(x) # removed: kernel_regularizer=kr, kernel_constraint=kc (and bias activated)
    # x = layers.LayerNormalization(epsilon=1e-5)(x); x = layers.ReLU()(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding, kernel_initializer=ki, 
                      use_bias=True, activation="relu")(x) # removed: kernel_regularizer=kr, kernel_constraint=kc (and bias activated)
    # x = layers.LayerNormalization(epsilon=1e-5)(x); x = layers.ReLU()(x) # ELU Akvitierungsfunktion replaced by ReLU
    return x

def unet3d(input_shape=(5,192,240,1), base_filters=16):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)            ; p1 = layers.MaxPooling3D((1,2,2))(c1)
    c2 = conv_block(p1, base_filters*2)              ; p2 = layers.MaxPooling3D((1,2,2))(c2)
    c3 = conv_block(p2, base_filters*4)              ; p3 = layers.MaxPooling3D((1,2,2))(c3)
    c4 = conv_block(p3, base_filters*8)              ; p4 = layers.MaxPooling3D((1,2,2))(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters*16)

    # Decoder
    u4 = layers.Conv3DTranspose(base_filters*8, (1,2,2), (1,2,2), padding="same")(bn)
    # Skip connection von c4 nach u4
    u4 = layers.Concatenate()([u4, c4])              ; c5 = conv_block(u4, base_filters*8)

    u3 = layers.Conv3DTranspose(base_filters*4, (1,2,2), (1,2,2), padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])              ; c6 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, (1,2,2), (1,2,2), padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])              ; c7 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters,   (1,2,2), (1,2,2), padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])              ; c8 = conv_block(u1, base_filters)

    out = layers.Conv3D(1, (1,1,1), activation="sigmoid",
                        kernel_initializer="he_normal")(c8)

    return models.Model(inputs=inputs, outputs=out, name=f"3D_U-Net_JENS_d4_bf{base_filters}_out_sigmoid")


# %%
# ==============================
# Loss & Metrics  (MAE + SSIM)
# ==============================

def _to_4d(yt, yp):
    # SSIM braucht 4D Tensoren: (B,D,H,W,C) -> (B*D,H,W,C)
    b,d,h,w,c = tf.unstack(tf.shape(yt))
    return tf.reshape(yt, (b*d, h, w, c)), tf.reshape(yp, (b*d, h, w, c))

def ssim_3d_mean_safe(y_true, y_pred, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    # SSIM in 3D via 2D-Umformung mit zusätzlichem clipping für Stabilität
    yt = clip01(y_true); yp = clip01(y_pred)
    yt4, yp4 = _to_4d(yt, yp)
    v = tf.image.ssim(yp4, yt4, max_val=max_val,
                      filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
    return tf.reduce_mean(v)

def ssim_metric(y_true, y_pred): return ssim_3d_mean_safe(y_true, y_pred, max_val=1.0)
    # ssim Metrik
ssim_metric.__name__ = "ssim" # logger name

def psnr_metric(y_true, y_pred):
    # PSNR Metrik
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)

psnr_metric.__name__ = "psnr" # logger name

class CombinedMAE_SSIM_Loss(tf.keras.losses.Loss):
    # Repliziert loss von Jens
    def __init__(self, alpha=0.7, name="mae_ssim"):
        super().__init__(name=name); self.alpha = float(alpha)
    def call(self, y_true, y_pred):
        yt = clip01(y_true); yp = clip01(y_pred)
        mae = tf.reduce_mean(tf.abs(yt - yp))
        ssimv = ssim_3d_mean_safe(yt, yp, max_val=1.0)
        return (1.0 - self.alpha) * mae + self.alpha * (1.0 - ssimv)



# %%
# ==============================
# Compile
# ==============================

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(optimizer=AdamW(learning_rate=1e-4),
    loss=tf.keras.losses.MeanAbsoluteError(), # loss=CombinedMAE_SSIM_Loss(alpha=0.7)
    metrics=[ssim_metric, tf.keras.metrics.MeanAbsoluteError(name="mae"), 
    tf.keras.metrics.MeanSquaredError(name="mse"), psnr_metric],
    jit_compile=False # Schon bei Imports deaktiviert, als Absicherung gedacht
)

# %%
# ==============================
# Callbacks + CSV_logging
# ==============================

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"

# Dictionary für callbacks, checkpoints, logger
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 200},
    "data_prep": {"size": 1, "group_len": None, "dtype": "float32"},
    "alpha": 0.7,
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
    code_name="unet_3d_JENS_V2",
    verbose_ckpt=1
)

# CSV logger
CSV_DIR = Path.home() / "data" / "logs_csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = CSV_DIR / f"{bf.code}_train_{stamp}.csv"

csv_cb = CSVLogger(filename=str(csv_path), separator=",", append=False)
# CSV-Logger zu den Callbacks packen
cbs = list(cbs) + [csv_cb]

# %%
# ==============================
# Training + kurze Evaluierung
# ==============================
print(">>> Phase 2: GPU training starts now!")

steps_per_epoch  = _steps(meta, "train", BATCH_SIZE)
validation_steps = _steps(meta, "val",   BATCH_SIZE)

if EPOCHS > 0:
    train_ds_rep = train_ds.repeat(EPOCHS)
    val_ds_rep   = val_ds.repeat(EPOCHS)

    model.fit(
        train_ds_rep,
        validation_data=val_ds_rep,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=0,
    )

    print(">>> Phase 3: Training complete!")
    final_val  = model.evaluate(val_ds,  return_dict=True, verbose=0)
    final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("FINAL VAL:",  {k: float(v) for k, v in final_val.items()})
    print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})
else:
    print(">>> Skipping main training (EPOCHS=0).")





# %%
# ==============================
# Mini-Sweep: Depth, Base-Filters, Out-Act, LR
# ==============================
# ----- Sweep-Settings -----
START_AT         = int(os.environ.get("START_AT", "1"))
CSV_DIR_SWEEP    = Path.home() / "data" / "logs_csv_scout_JENS_V2"
CKPT_ROOT_SWEEP  = Path.home() / "data" / "checkpoints_3d_unet_scout_JENS_V2"
CSV_DIR_SWEEP.mkdir(parents=True, exist_ok=True)
CKPT_ROOT_SWEEP.mkdir(parents=True, exist_ok=True)

# Sweep-Parameter
DEPTHS         = [4, 5]
BASE_FILTERS   = [24, 32] # 8 removed
OUT_ACTS       = ["sigmoid"]
LEARNING_RATES = [3e-4, 1e-5] # 1e-4, 3e-5 removed
BATCH_SIZES    = [32] # 128 removed
MAX_EPOCHS_SCOUT = 20


def _safe_tag(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", s)
    if not re.match(r"^[A-Za-z0-9.]", t): t = "A" + t
    return t

def build_unet3d(input_shape, base_filters=16, depth=4, output_activation="sigmoid", name=None):
    """
    U-Net 3D mit variabler Tiefe. Erwartet 5D-Input (D,H,W in den letzten 4 Achsen),
    pooled/transponiert nur über (H,W) → Zeit/Slices (D) bleiben erhalten.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []
    f = base_filters

    # Encoder
    for _ in range(depth):
        x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
        x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
        skips.append(x)
        x = layers.MaxPooling3D(pool_size=(1,2,2))(x)   # nur H,W
        f *= 2

    # Bottleneck
    x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
    x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)

    # Decoder
    for skip in reversed(skips):
        f //= 2
        x = layers.Conv3DTranspose(f, (1,2,2), strides=(1,2,2), padding="same",
                                   kernel_initializer="he_normal", activation="relu")(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)
        x = layers.Conv3D(f, (3,3,3), padding="same", kernel_initializer="he_normal", activation="relu")(x)

    out = layers.Conv3D(1, (1,1,1), activation=output_activation, kernel_initializer="he_normal")(x)
    return models.Model(inputs=inputs, outputs=out, name=(name or f"UNet3D_d{depth}_bf{base_filters}_out{output_activation}"))


def _build_model(input_shape, base_filters, depth, out_act, lr, name):
    m = build_unet3d(input_shape, base_filters=base_filters, depth=depth,
                     output_activation=out_act, name=name)
    m.compile(
        optimizer=AdamW(learning_rate=lr), # AdamW nicht identisch zu JENS
        loss=tf.keras.losses.MeanAbsoluteError(),   # <- nur MAE
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            psnr_metric,
        ],
        jit_compile=False
    )
    return m


def _prepare_callbacks(tag, batch_size):
    run_meta = {
        "batch_size": batch_size,
        "epochs": MAX_EPOCHS_SCOUT,
        "early_stopping": None,
        "data_prep": {"size": 1, "group_len": None, "dtype": "float32"},
        "alpha": 0.7, "loss_components": {"mae": 0.3, "ssim": 0.7},
    }
    cbs, _, _ = build_standard_callbacks(
        ckpt_root=CKPT_ROOT_SWEEP,
        run_meta=run_meta,
        monitor="val_loss",
        patience_es=10**9,            # faktisch: kein ES im Scout
        reduce_on_plateau=False,      # kurz, konstant trainieren
        include_nan_guards=True,
        include_logger=True,
        code_name=f"unet3d_scout_{tag}",
        verbose_ckpt=0
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = CSV_DIR_SWEEP / f"{tag}_{stamp}.csv"
    csv_cb = CSVLogger(str(csv_path), separator=",", append=False)
    return list(cbs) + [csv_cb]

def run_mini_sweep():
    runs = 0

    for bs in BATCH_SIZES:
        # Datasets EINMAL fuer dieses bs (Augment->Normalize bereits in pipeline_*)
        train_ds, val_ds, test_ds, meta = build_1stack_datasets_flat(
            data_dir=Path.home() / "data" / "original_data",
            batch_train=bs, batch_eval=bs,
            preproc_train=pipeline_train,
            preproc_eval=pipeline_val,
            augmenter=None,
            deterministic=False,
            cache_after_preproc=False
        )
        spe  = _steps(meta, "train", bs)
        vste = _steps(meta, "val",   bs)
        tste = _steps(meta, "test",  bs)

        for depth in DEPTHS:
            for bf in BASE_FILTERS:
                for outa in OUT_ACTS:
                    for lr in LEARNING_RATES:
                        runs += 1
                        if runs < START_AT:
                            continue

                        raw_tag = f"sweep{runs}_d{depth}_bf{bf}_out{outa}_lr{lr:g}_bs{bs}"
                        tag = _safe_tag(raw_tag)
                        print(f"\n[SWEEP] {raw_tag}")

                        callbacks_run = _prepare_callbacks(tag, bs)
                        model = _build_model(meta["input_shape"], bf, depth, outa, lr, name=f"Scout_{tag}")

                        # Einfacher Fit (kein Rebatch/Retry; fuege deinen Retry-Block wieder ein, falls du CUDNN-Scratch-Fehler kennst)
                        model.fit(
                            train_ds.repeat(MAX_EPOCHS_SCOUT),
                            validation_data=val_ds.repeat(MAX_EPOCHS_SCOUT),
                            epochs=MAX_EPOCHS_SCOUT,
                            steps_per_epoch=spe,
                            validation_steps=vste,
                            callbacks=callbacks_run,
                            verbose=0,
                        )
                        _ = model.evaluate(val_ds,  steps=vste, return_dict=True, verbose=0)
                        _ = model.evaluate(test_ds, steps=tste, return_dict=True, verbose=0)

    print(f"\n[SWEEP] Completed {runs} runs. CSVs in {CSV_DIR_SWEEP}")

run_mini_sweep()
