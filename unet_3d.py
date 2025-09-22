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

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") # Increases performance without loss of quality (calculations still done with float_32 precision)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from unet_3d_data import prepare_in_memory_5to5
from pathlib import Path

import uuid # for naming files and callbacks


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

print(">>> Phase 1: Starting data prep on CPU...")
(results, size) = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    use_vst=False, # No anscombe transform
    size=5,
    group_len=41,
    dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]


INPUT_SHAPE = X_train.shape[1:]  # (5, H, W, 1)

# %%
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

    outputs = layers.Conv3D(1, (1,1,1), dtype="float32", activation="sigmoid")(c6)
    return models.Model(inputs, outputs, name="3D_U-Net")



# %%
# =========== Defining Loss function MAE + MS-SSIM (slice-wise) ========

ALPHA = 0.7  # Weight for MS-SSIM

def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    """
    Generates deterministic matrix and samples indices using highest values per row
    """
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0]) # (B,D) matrix with random values
    topk = tf.math.top_k(rnd, k=k).indices                                 # Search for 2 highest values per row
    return topk

def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    """
    Defining MS-SSIM for the loss function equivalently as in the paper
    """
    # y: (B, D, H, W, C)
    batch_size = tf.shape(y_true)[0]
    depth = tf.shape(y_true)[1]                               # Number of 2D slices
    idx = _sample_depth_indices(batch_size, depth, k=k)       # (B,k)
    # gathering chosen slices
    y_groundtruth = tf.gather(y_true, idx, batch_dims=1)      # (B,k,H,W,C)
    y_model = tf.gather(y_pred, idx, batch_dims=1)            # (B,k,H,W,C)
    # flatten to 2D pictures
    y_groundtruth_2 = tf.reshape(y_groundtruth, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_model_2 = tf.reshape(y_model, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    ms  = tf.image.ssim_multiscale(y_groundtruth_2, y_model_2, max_val=1.0)     # (B*k,)
    return 1.0 - tf.reduce_mean(ms)

def combined_loss(y_true, y_pred, k_slices=1):
    """
    Combining the loss composite of MAE and MS-SSIM
    (MAE stable and useful for strong signals --> Bragg peaks)
    (MS-SSIM focuses on structure --> CDW satellite signals)
    """
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss_sampled(y_true, y_pred, k=k_slices)  # K=1
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms

def ms_ssim_metric(y_true, y_pred):
    """
    Showing MS-SSIM metric during training
    """
    yt2 = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))

def psnr_metric(y_true, y_pred):
    """
    Showing PSNR metric during training
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)



# %%
# ======== Naming files and making callbacks =======

def parse_filename_simple(name: str):
    """
    Expects names like:
      - TMP_<uuid>_valloss_0.0123_PSNR_32.1.keras
      - V1_valloss_0.0119_PSNR_32.5.keras
      - V2_valloss_0.013.keras_PSNR_31.4.keras
    """
    if not name.endswith(".keras"):
        return None

    base = name[:-6]                   # remove ".keras"
    parts = base.split("_")

    # Assert that valloss is present
    if "valloss" not in parts:
        return None

    val_loss = None
    psnr = None
    for i, p in enumerate(parts):
        if p == "valloss" and i + 1 < len(parts):
            try:
                val_loss = float(parts[i + 1])
            except ValueError:
                return None
        if p == "PSNR" and i + 1 < len(parts):
            try:
                psnr = float(parts[i + 1])
            except ValueError:
                psnr = None

    if val_loss is None:
        return None
    return {"val_loss": val_loss, "psnr": psnr}


def rank_and_rename(root: Path):
    """
    Reads all .keras files in folder and sorts them according to val_loss
    Renames them suchn that V1 has the lowest val_loss, V2 the second lowest, etc.
    Finale names: V{rank}_valloss_{:.3g}[_PSNR_{:.3g}].keras
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    items = []
    for p in root.iterdir():
        if p.is_file() and p.suffix == ".keras":
            m = parse_filename_simple(p.name)
            if m is not None:
                items.append((p, m["val_loss"], m["psnr"]))

    if not items:
        return
    
    # Smallest val_loss first
    items.sort(key=lambda x: x[1])

    # Avoiding collisions: two-step renaming(first temp, then final)
    temp_paths = []
    for path, vloss, psnr in items:
        tmp = root / f".tmp_{uuid.uuid4().hex}.keras"
        os.replace(path, tmp)
        temp_paths.append((tmp, vloss, psnr))

    # Final naming
    for rank, (tmp, vloss, psnr) in enumerate(temp_paths, start=1):
        vloss_str = f"{vloss:.3g}"
        psnr_part = f"_PSNR_{psnr:.3g}" if (psnr is not None) else ""
        final = root / f"V{rank}_valloss_{vloss_str}{psnr_part}.keras"
        os.replace(tmp, final)


class RankRenameCallback(callbacks.Callback):
    """Fuehrt am Trainingsende das globale Ranking & Umbenennen aus."""
    def __init__(self, root: Path):
        super().__init__()
        self.root = Path(root)

    def on_train_end(self, logs=None):
        rank_and_rename(self.root)


def file_names(root: Path):
    """
    Liefert (ckpt_best, rank_cb).

    - ckpt_best speichert waehrend des Runs genau EINE Datei (save_best_only=True)
      mit temporaerem Namen:
        TMP_<uuid>_valloss_{val_loss:.3g}_PSNR_{psnr_metric:.3g}.keras
    - rank_cb sortiert am Ende alle .keras Dateien im Ordner und
      vergibt finale Namen:
        V1_valloss_{:.3g}_PSNR_{:.3g}.keras, V2_..., V3_...
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # WICHTIG: Platzhalter muss zur Metrik in model.compile passen -> psnr_metric
    tmp_name = (
        f"TMP_{uuid.uuid4().hex}_valloss_" + "{val_loss:.3g}" +
        "_PSNR_" + "{psnr_metric:.3g}.keras"
    )
    filepath = root / tmp_name

    ckpt_best = callbacks.ModelCheckpoint(
        filepath=str(filepath),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    return ckpt_best, RankRenameCallback(root)


# %%
# ======== Train =======

print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=combined_loss,
    metrics=["mae", "mse", psnr_metric, ms_ssim_metric],
    jit_compile=False # Would be false per default, but just to be sure
)
# model.summary()

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
ckpt_cb, rank_cb = file_names(ckpt_root)

cbs = [
    ckpt_cb,  # speichert das beste Modell dieses Runs (temp. Name)
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    rank_cb,  # benennt am Ende alles in V1/V2/... um
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=2
)
print(">>> Training complete")

