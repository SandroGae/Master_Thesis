# unet_3d_JENS_V2_2D.py
# ==============================
# 0) Imports & global setup
# ==============================
#!/usr/bin/env python3
import tensorflow as tf
print("[0] start", flush=True)
tf.config.optimizer.set_jit(False)  # XLA JIT aus!!!

# Sichtbare GPUs loggen und Growth aktivieren (kein hartes VRAM-Limit setzen)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs sichtbar:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("WARN: Keine GPU sichtbar –> läuft auf CPU.")

from pathlib import Path
print("[1] bli", flush=True)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler
print("[2] bla", flush=True)
from datetime import datetime
import math

from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_1stack_datasets_flat, clip01, build_standard_callbacks
print("[3] blub", flush=True)

# %%
# ==============================
# Model of Jens
# ==============================
def VDSR(input_shape, filters=64, kernel_initializer='he_normal'):
    """VDSR model architecture (Very Deep Super-Resolution Neural Network).

    - 'he_normal' weights initializer
    - 64 filters per layer
    - 20 convolutional layers
    - parametric rectifying linear unit (PReLU) as activation

    Reference: 
    J. Kim, J. K. Lee, and K. M. Lee, 
    “Accurate Image Super-Resolution Using Very Deep Convolutional Networks,” 
    in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2016, pp. 1646–1654. 
    doi: 10.1109/CVPR.2016.182.

    Parameters
    ----------
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    filters : int
        Number of filters per layer
    kernel_initializer : string
        Kernel initializer to be used as defined by keras.initializers
    
    Returns
    -------
    keras.Model
    """

    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                               kernel_initializer=kernel_initializer)(inp)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)   # echte Layer, mit alpha-Variable

    for _ in range(19):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                   kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    out = tf.keras.layers.Conv2D(1, 3, padding='same',
                                 kernel_initializer=kernel_initializer)(x)
    return tf.keras.Model(inp, out, name="VDSR")

    """
    # Initialize a parametric linear rectifier unit
    para_relu = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.constant(0.25))

    # Create the neural network
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, activation=para_relu, kernel_initializer=kernel_initializer, padding='same') (input)

    for _ in range(19):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, kernel_initializer=kernel_initializer, padding='same') (x)
        x = tf.keras.layers.Activation(para_relu) (x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=kernel_initializer, padding='same') (x)
    model = tf.keras.Model(input, x, name="VDSR")

    return model
    """


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


BATCH_SIZE = 8
EPOCHS = 100

def augment_fliplr_only(x, y):
    # Augmentation: nur Left-Right wie bei Jens
    do_lr = tf.random.uniform(()) < 0.5
    fliplr = lambda t: tf.reverse(t, axis=[2])  # 4D: (D,H,W,C) -> W ist axis 2 (for 5D: (B,D,H,W,C) -> W is axis 3!!!)
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

reset_random_seeds(0)


train_ds, val_ds, test_ds, meta = build_1stack_datasets_flat(
    data_dir=Path.home() / "data" / "original_data",
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    read_block=128,
    preproc_train=pipeline_train,
    preproc_eval=pipeline_val,
    out_rank=4,
    cache_after_preproc=False,
)


INPUT_SHAPE = meta["input_shape"]
print(">>> Datasets created (D =", meta["D"], ")")

def _steps(meta, split, batch):
    N = {"train": meta["n_train"], "val": meta["n_val"], "test": meta["n_test"]}[split]
    return math.ceil(N / batch)


# %%
# ==============================
# PSNR metric
# ==============================

def psnr_metric(y_true, y_pred):
    # PSNR Metrik
    yt = clip01(y_true); yp = clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)

psnr_metric.__name__ = "psnr" # logger name



# %%
# ==============================
# Compile
# ==============================


model = VDSR(input_shape=INPUT_SHAPE)
model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True), metrics=['mae'])


# %%
# ==============================
# Callbacks + CSV_logging
# ==============================

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"

# Dictionary für callbacks, checkpoints, logger
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    # "early_stopping": {"monitor": "val_loss", "patience": None},
    "data_prep": {"size": 1, "group_len": None, "dtype": "float32"},
    # "alpha": 0.3,
    "loss_components": {"mae": 1.0}  # einfache MAE
}

cbs, bf, ckpt_best = build_standard_callbacks(
    ckpt_root=ckpt_root,
    run_meta=run_meta,
    monitor="val_loss",
    patience_es=200,
    reduce_on_plateau=False,
    reduce_factor=1,
    reduce_patience=100,
    min_lr=1e-6,
    include_nan_guards=True,
    include_logger=True,
    code_name="unet_3d_JENS_V2",
    verbose_ckpt=1
)

# EarlyStopping komplett entfernen, damit es wirklich Jens-like ist
cbs = [cb for cb in cbs if not isinstance(cb, EarlyStopping)]

# CSV logger
CSV_DIR = Path.home() / "data" / "logs_csv"

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = CSV_DIR / f"{bf.code}_train_{stamp}.csv"

csv_cb = CSVLogger(filename=str(csv_path), separator=",", append=False)
# CSV-Logger zu den Callbacks packen
cbs = list(cbs) + [csv_cb]



# %%
# ==============================
# Training
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





