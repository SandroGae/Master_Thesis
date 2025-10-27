# unet_3d_JENS_V2.py
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
from train_utils import build_standard_callbacks, build_5stack_datasets_grouped, clip01


# Reproduzierbarkeit
seed = 0
reset_random_seeds(seed)
AUTO = tf.data.AUTOTUNE # tensorflow wählt selbst wie viele elemente parallel geladen/verarbeitet werden

# %%
# ==============================
# 1–3) Daten-Streaming (kein RAM-Fullload)
# ==============================

# Normalisierung identisch zu Jens
preproc_train_slice = SumScaleNormalizer(
    scale_min=5000, 
    scale_max=15000,
    pre_offset=0.0,
    normalize_label=True,
    batch_mode=False   # 4D Einzel-Sample (D,H,W,C)
)
preproc_valid_slice = SumScaleNormalizer(
    scale_min=10000, 
    scale_max=10001,
    pre_offset=0.0,
    normalize_label=True,
    batch_mode=False   # 4D Einzel-Sample
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

def augment_5stack_flips(x, y):
    # Spiegelt x und y zufällig links-rechts
    do_lr = tf.random.uniform(()) < 0.5
    fliplr = lambda t: tf.reverse(t, axis=[2])  # 4D: (D,H,W,C) -> W ist axis 2
    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    return x, y

print(">>> Phase 1: Baue Datensatz via train_utils...")
DATA_DIR = Path.home() / "data" / "original_data"
train_ds, val_ds, test_ds, meta = build_5stack_datasets_grouped(
    data_dir=DATA_DIR,
    group_len=41,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    preproc_train=map_slice_wise(preproc_train_slice),
    preproc_eval=map_slice_wise(preproc_valid_slice),
    augmenter=augment_5stack_flips,
    deterministic=False, # False: Map und Prefetch dürfen in zufälliger Reihenfolge arbeiten
    cache_after_preproc=False, # False: Augmentation wird bei jeder Epoche neu berechnet
)

def _steps(meta, split, batch):
    # Berechnet Anzahl Batches pro Epoche = Anzahl Updates pro Epoche
    spp = meta["samples_per_group"] # 37 Trainingsdaten pro 41er gruppe
    groups = meta[f"{split}_groups"]
    total = groups * spp
    return math.ceil(total / batch)


INPUT_SHAPE = meta["input_shape"]
print(">>> Datasets created")


# %%
# ==============================
# 4) Model Architektur (Depth=4)
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
# 5) Loss & Metrics  (MAE + SSIM)
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
# 6) Compile
# ==============================

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(optimizer=AdamW(learning_rate=1e-4),
    loss=tf.keras.losses.MeanAbsoluteError(name="mae_loss"),
    metrics=[ssim_metric, tf.keras.metrics.MeanAbsoluteError(name="mae"), 
    tf.keras.metrics.MeanSquaredError(name="mse"), psnr_metric],
    jit_compile=False # Schon bei Imports deaktiviert, als Absicherung gedacht
)

# %%
# ==============================
# 7) Callbacks + CSV_logging
# ==============================

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"

# Dictionary für callbacks, checkpoints, logger
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 200},
    "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
    # "alpha": 0.7,
    # "loss_components": {"mae": 0.3, "ssim": 0.7}  # einfache SSIM
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
# 8) Training + kurze Evaluierung
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
# 11) Mini-Sweep: Tiefe, Base-Filters, Output-Activation, Loss GPTs WORK!
# ==============================
START_AT = int(os.environ.get("START_AT", "1"))  # Bei welchem eun beginnen?

CSV_DIR_SWEEP = Path.home() / "data" / "logs_csv_scout_JENS_V2"
CSV_DIR_SWEEP.mkdir(parents=True, exist_ok=True)

def _safe_tag(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", s)
    if not re.match(r"^[A-Za-z0-9.]", t):
        t = "A" + t
    return t

def build_unet3d(input_shape, base_filters=16, depth=3,
                 output_activation="sigmoid", name=None):
    """Erstellt eine 3D-U-Net-Architektur mit variabler Tiefe."""
    inputs = layers.Input(shape=input_shape)
    skips = []
    x = inputs
    filters = base_filters

    # ----- Encoder -----
    for _ in range(depth):
        x = conv_block(x, filters)
        skips.append(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
        filters *= 2

    # ----- Bottleneck -----
    x = conv_block(x, filters)

    # ----- Decoder -----
    for skip in reversed(skips):
        filters //= 2
        x = layers.Conv3DTranspose(filters, (1, 2, 2), strides=(1, 2, 2),
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   activation="relu")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)

    # ----- Output -----
    out = layers.Conv3D(1, (1, 1, 1),
                        activation=output_activation,
                        kernel_initializer="he_normal")(x)

    model_name = name or f"UNet3D_d{depth}_bf{base_filters}_out{output_activation}"
    return models.Model(inputs=inputs, outputs=out, name=(name or f"UNet3D_d{depth}_bf{base_filters}_out{output_activation}"))


# Sweep-Parameter
DEPTHS         = [3, 4]
BASE_FILTERS   = [8, 16] # 24 removed
OUT_ACTS       = ["sigmoid"]
LEARNING_RATES = [3e-4, 3e-5] # 1e-4 removed
BATCH_SIZES    = [32] # 128 removed
MAX_EPOCHS_SCOUT = 20

class InjectStatic(Callback):
    def __init__(self, **static):
        super().__init__()
        self.static = static

    @staticmethod
    def _is_number(x):
        import numpy as np
        return isinstance(x, (int, float, np.integer, np.floating))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for k, v in self.static.items():
            if self._is_number(v):
                logs[k] = float(v)
        if "run_tag" in self.static and "run_tag_hash" not in logs:
            h = abs(hash(str(self.static["run_tag"]))) % (10**9)
            logs["run_tag_hash"] = float(h)
        if "out_act" in self.static and "out_act_id" not in logs:
            act_map = {"sigmoid": 1.0, "tanh": 2.0, "linear": 3.0}
            logs["out_act_id"] = float(act_map.get(str(self.static["out_act"]).lower(), 0.0))

def run_mini_sweep():
    runs = 0
    ckpt_root_scout = Path.home() / "data" / "checkpoints_3d_unet_scout_JENS_V2"
    CSV_DIR_SWEEP = Path.home() / "data" / "logs_csv_scout_JENS_V2"
    CSV_DIR_SWEEP.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path.home() / "data" / "original_data"

    for bs in BATCH_SIZES:
        # ---- Datasets EINMAL pro "nominalem" bs ----
        train_ds_bs, val_ds_bs, test_ds_bs, meta_bs = build_5stack_datasets_grouped(
            data_dir=DATA_DIR,
            group_len=41,
            batch_train=bs,
            batch_eval=bs,
            preproc_train=map_slice_wise(preproc_train_slice),
            preproc_eval=map_slice_wise(preproc_valid_slice),
            augmenter=augment_5stack_flips,
            deterministic=False,
            cache_after_preproc=False
        )

        spe_bs  = _steps(meta_bs, "train", bs)
        vste_bs = _steps(meta_bs, "val",   bs)
        tste_bs = _steps(meta_bs, "test",  bs)

        for depth in DEPTHS:
            for bf in BASE_FILTERS:
                for outa in OUT_ACTS:
                    for lr in LEARNING_RATES:
                        runs += 1
                        if runs < START_AT:
                            continue  # vorangehende Runs ueberspringen

                        raw_tag = f"d{depth}_bf{bf}_ReLU_LN_out{outa}_lr{lr:g}_bs{bs}__mae+ssim:0.7"
                        tag = _safe_tag(raw_tag)

                        # ---- Basiswahl eff_bs (schwere Configs runtersetzen) ----
                        eff_bs = bs
                        if bf >= 24 and bs == 128:
                            eff_bs = 32

                        # ---- Callbacks ----
                        run_meta_scout = {
                            "batch_size": eff_bs,  # effektive Trainings-Batch
                            "epochs": MAX_EPOCHS_SCOUT,
                            "early_stopping": {"monitor": "val_loss", "patience": 10**9},
                            "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
                            # "alpha": 0.7, "loss_components": {"mae": 0.3, "ssim": 0.7}
                        }
                        cbs_scout, _, _ = build_standard_callbacks(
                            ckpt_root=ckpt_root_scout,
                            run_meta=run_meta_scout,
                            monitor="val_loss",
                            patience_es=10**9,
                            reduce_on_plateau=False,
                            include_nan_guards=True,
                            include_logger=True,
                            code_name=f"unet_3d_JENS_V2_sweep{runs}_{tag}_3D",
                            verbose_ckpt=0
                        )
                        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        csv_path = CSV_DIR_SWEEP / f"sweep{runs}_{tag}_{stamp}.csv"
                        csv_cb = CSVLogger(str(csv_path), separator=",", append=False)
                        inject = InjectStatic(run_tag=tag, depth=depth, base_filters=bf,
                                              out_act=outa, lr=lr, batch_size=eff_bs)
                        cbs_scout = [inject] + list(cbs_scout) + [csv_cb]

                        print(f"\n[SWEEP] Run {runs}: {raw_tag}"
                              + ("" if eff_bs == bs else f"  (eff_bs={eff_bs} statt {bs})"))

                        # ---------- Hilfsfunktionen ----------
                        def _make_datasets(eff_batch):
                            if eff_batch == bs:
                                tr_fit = train_ds_bs.repeat(MAX_EPOCHS_SCOUT)
                                va_fit = val_ds_bs.repeat(MAX_EPOCHS_SCOUT)
                                use_val  = val_ds_bs
                                use_test = test_ds_bs
                                spe_run, vste_run, tste_run = spe_bs, vste_bs, tste_bs
                            else:
                                tr_fit = train_ds_bs.unbatch().batch(eff_batch).repeat(MAX_EPOCHS_SCOUT)
                                va_fit = val_ds_bs.unbatch().batch(eff_batch).repeat(MAX_EPOCHS_SCOUT)
                                use_val  = val_ds_bs.unbatch().batch(eff_batch)
                                use_test = test_ds_bs.unbatch().batch(eff_batch)
                                spe_run  = _steps(meta_bs, "train", eff_batch)
                                vste_run = _steps(meta_bs, "val",   eff_batch)
                                tste_run = _steps(meta_bs, "test",  eff_batch)
                            return tr_fit, va_fit, use_val, use_test, spe_run, vste_run, tste_run

                        def _build_model():
                            m = build_unet3d(
                                meta_bs["input_shape"],
                                base_filters=bf,
                                depth=depth,
                                output_activation=outa,
                                name=f"Scout_{tag}"
                            )
                            m.compile(optimizer=AdamW(learning_rate=lr),
                                loss=tf.keras.losses.MeanAbsoluteError(name="mae_loss"),
                                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), # SSIM removed
                                        tf.keras.metrics.MeanSquaredError(name="mse"),
                                        psnr_metric],
                                jit_compile=False
                            )
                            return m


                        # ---------- Train mit Retry bei cuDNN-Workspace-Fehler ----------
                        current_bs = eff_bs
                        while True:
                            (train_ds_fit, val_ds_fit, use_val_ds, use_test_ds,
                             spe_run, vste_run, tste_run) = _make_datasets(current_bs)

                            model = _build_model()
                            try:
                                model.fit(
                                    train_ds_fit,
                                    validation_data=val_ds_fit,
                                    epochs=MAX_EPOCHS_SCOUT,
                                    steps_per_epoch=spe_run,
                                    validation_steps=vste_run,
                                    callbacks=cbs_scout,
                                    verbose=0,
                                )
                                _ = model.evaluate(use_val_ds,  steps=vste_run, return_dict=True, verbose=0)
                                _ = model.evaluate(use_test_ds, steps=tste_run, return_dict=True, verbose=0)
                                break  # Erfolg
                            except Exception as e:
                                msg = str(e)
                                needs_retry = (
                                    "CUDNN failed to allocate the scratch space" in msg
                                    or "scratch space" in msg.lower()
                                    or "cudnn" in msg.lower()
                                )
                                tf.keras.backend.clear_session()
                                import gc; gc.collect()
                                if needs_retry and current_bs > 8:
                                    current_bs = max(8, current_bs // 2)
                                    print(f"   -> cuDNN-Workspace-Problem, retry mit eff_bs={current_bs}")
                                    continue
                                else:
                                    raise

    print(f"\n[SWEEP] Completed {runs} runs. CSVs in {CSV_DIR_SWEEP}")

run_mini_sweep()

