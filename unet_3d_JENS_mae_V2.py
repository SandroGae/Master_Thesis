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
# unet_3d_JENS_mae.py
# ==============================
# 0) Imports & global setup
# ==============================

from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(False)

from tensorflow.keras import regularizers, constraints, layers, models, callbacks
from tensorflow.keras.optimizers import AdamW

from unet_3d_data_JENS import prepare_in_memory_5to5
from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_standard_callbacks, clip01


seed = 0
reset_random_seeds(seed)

for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# %%
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
EPOCHS     = 200
BATCH_SIZE = 32

# %%
# ==============================
# 2) Preprocessing & Augment
# ==============================

preproc_train_slice = SumScaleNormalizer(
    scale_range=[5000, 15001], pre_offset=0.0, normalize_label=True,
    axis=(1,2,3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0.,1.]
)
preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001], pre_offset=0.0, normalize_label=True,
    axis=(1,2,3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0.,1.]
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
    def fliplr(t): return tf.reverse(t, axis=[2])
    def flipud(t): return tf.reverse(t, axis=[1])
    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    x = tf.cond(do_ud, lambda: flipud(x), lambda: x)
    y = tf.cond(do_ud, lambda: flipud(y), lambda: y)
    return x, y


# %%
# ==============================
# 3) Datasets
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
        ds = ds.map(lambda x,y: tuple(preproc(x,y)), num_parallel_calls=num_calls)
    if cache_in_memory: 
        ds = ds.cache()
    if augmenter is not None:
        ds = ds.map(lambda x,y: augmenter(x,y), num_parallel_calls=num_calls)
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
                   augmenter=None, check_nans=True, prefetch_n=1, num_calls=1)
test_ds  = make_ds(X_test, Y_test, shuffle=False,
                   preproc=map_slice_wise(preproc_valid_slice),
                   augmenter=None, check_nans=True, prefetch_n=1, num_calls=1)
print(">>> Datasets created")


# %%
# ==============================
# 4) Model
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
        name=f"UNet3D_JENS_mae_d4_bf{base_filters}_out{output_activation}"
    )



# %%
# ==============================
# 5) Loss & Metrics (MAE)
# ==============================

def mae_loss(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.reduce_mean(tf.abs(yt - yp))

def psnr_metric(y_true, y_pred):
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)
psnr_metric.__name__ = "psnr"

# %%
# ==============================
# 6) Compile
# ==============================

opt = AdamW(learning_rate=1e-5, epsilon=1e-3, global_clipnorm=0.1, weight_decay=0.0, amsgrad=False)
model = unet3d(input_shape=INPUT_SHAPE, base_filters=16, output_activation="tanh")
model.compile(
    optimizer=opt,
    loss=mae_loss,
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
             tf.keras.metrics.MeanSquaredError(name="mse"),
             psnr_metric],
    jit_compile=False
)

# %%
# ==============================
# 7) Callbacks (gemeinsam)
# ==============================

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor": "val_loss", "patience": 20},
    "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
    "alpha": None,                 # kein Kombi-Loss
    "loss_components": {"mae": 1.0}
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
    code_name="unet_3d_JENS_mae",
    verbose_ckpt=1
)

# %%
# ==============================
# 8) Train & Evaluate
# ==============================

print(">>> Phase 3: GPU training starts now!")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=0)
print(">>> Phase 3: Training complete!")

final_val = model.evaluate(val_ds, return_dict=True, verbose=0)
print("FINAL VAL:", {k: float(v) for k, v in final_val.items()})
final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})

# %%
"""
# ==============================
# 9) Mini-Sweep (MAE): Depth, Base-Filters, Output-Aktivierung
#      -> je Run max. 10 Epochen, JSON-Report, sauberer Logger
# ==============================

import json, datetime, re
from itertools import product

EVAL_DIR = Path.home() / "data" / "model_evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def _safe_tag(s: str) -> str:
    # Nur A-Za-z0-9_.\\/>- erlauben; Rest -> '_'.
    t = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", s)
    if not re.match(r"^[A-Za-z0-9.]", t):
        t = "A" + t
    return t

# U-Net mit variabler Tiefe (Layer/Norm/Aktivierung identisch zu deinem conv_block)
def build_unet3d_depth(input_shape,
                       *, base_filters=16, depth_levels=3,
                       output_activation="sigmoid",
                       name=None):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    skips = []
    x = inputs
    filters = base_filters
    for _ in range(depth_levels):
        x = conv_block(x, filters)                  # exakt dein Block: LN+ELU
        skips.append(x)
        x = layers.MaxPooling3D((1,2,2))(x)
        filters *= 2
    # Bottleneck
    x = conv_block(x, filters)
    # Decoder
    for level in reversed(range(depth_levels)):
        filters //= 2
        x = layers.Conv3DTranspose(filters, (1,2,2), (1,2,2), padding="same")(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block(x, filters)
    out = layers.Conv3D(1, (1,1,1),
                        activation=output_activation,
                        kernel_initializer="glorot_uniform")(x)
    model_name = name or f"UNet3D_d{depth_levels}_bf{base_filters}_out{output_activation}"
    model_name = _safe_tag(model_name)
    return models.Model(inputs, out, name=model_name)

# Suchraum
DEPTHS      = [2, 3, 4]
BASE_FILTER = [8, 16, 24]
OUT_ACTS    = ["sigmoid", "tanh", "linear"]

MAX_EPOCHS_SCOUT = 10

def run_mini_sweep_mae():
    runs = 0
    ckpt_root_scout = Path.home() / "data" / "checkpoints_3d_unet_scout_mae"

    for depth, bf, outa in product(DEPTHS, BASE_FILTER, OUT_ACTS):
        runs += 1
        raw_tag = f"d{depth}_bf{bf}_ELU_LN_out{outa}__mae"
        tag = _safe_tag(raw_tag)

        # Modell
        model = build_unet3d_depth(INPUT_SHAPE,
                                   base_filters=bf,
                                   depth_levels=depth,
                                   output_activation=outa,
                                   name=f"Scout_{tag}")

        # Compile (MAE wie im Hauptmodell)
        opt_scout = AdamW(learning_rate=1e-4, epsilon=1e-3,
                          global_clipnorm=0.0, weight_decay=0.0, amsgrad=False)
        model.compile(
            optimizer=opt_scout,
            loss=mae_loss,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                     tf.keras.metrics.MeanSquaredError(name="mse"),
                     psnr_metric],
            jit_compile=False
        )

        # Callbacks/Logger (dein Style)
        run_meta_scout = {
            "batch_size": BATCH_SIZE,
            "epochs": MAX_EPOCHS_SCOUT,
            "early_stopping": {"monitor": "val_loss", "patience": 3},
            "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
            "alpha": None,
            "loss_components": {"mae": 1.0}
        }
        cbs_scout, _, _ = build_standard_callbacks(
            ckpt_root=ckpt_root_scout,
            run_meta=run_meta_scout,
            monitor="val_loss",
            patience_es=3,
            reduce_on_plateau=True,
            reduce_factor=0.5,
            reduce_patience=2,
            min_lr=1e-6,
            include_nan_guards=True,
            include_logger=True,
            code_name=f"sweep_mae_{tag}",
            verbose_ckpt=0
        )

        print(f"\n[SWEEP] Run {runs}: {raw_tag}")

        # Train (einmal, mit Logger) – 10 Epochen max, early stopping greift ggf. frueher
        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=MAX_EPOCHS_SCOUT, callbacks=cbs_scout, verbose=0
        )

        # Eval
        val_res  = model.evaluate(val_ds,  return_dict=True, verbose=0)
        test_res = model.evaluate(test_ds, return_dict=True, verbose=0)

        # JSON schreiben
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = {
            "timestamp": stamp,
            "run_tag": raw_tag,
            "run_tag_safe": tag,
            "model_name": model.name,
            "config": {
                "depth_levels": depth,
                "base_filters": bf,
                "norm": "LN", "act": "ELU",
                "output_activation": outa
            },
            "loss_name": "mae",
            "epochs_trained": int(len(history.history.get("loss", []))),
            "final_val":  {k: float(v) for k, v in val_res.items()},
            "final_test": {k: float(v) for k, v in test_res.items()}
        }
        vloss = out["final_val"].get("loss", None)
        vpsnr = out["final_val"].get("psnr", None)
        fname_core = f"sweep_mae_{stamp}_{tag}"
        fname = (
            f"{fname_core}_valloss_{vloss:.4e}_valpsnr_{vpsnr:.2f}.json"
            if (vloss is not None and vpsnr is not None) else
            f"{fname_core}.json"
        )
        with open(EVAL_DIR / fname, "w") as f:
            json.dump(out, f, indent=2)

        # Speicher freigeben
        tf.keras.backend.clear_session()

    print(f"\n[SWEEP] Completed {runs} runs. JSONs in {EVAL_DIR}")

# Falls du direkt nach dem Null-Training loslegen willst:
# EPOCHS = 0 oben setzen, bestehendes fit laeuft ohne Training,
# danach den Sweep starten:
run_mini_sweep_mae()
"""
