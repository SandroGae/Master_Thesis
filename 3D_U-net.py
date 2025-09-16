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
# ======== Imports + GPU memory growth =======
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Allocate GPU memory dynamically as needed
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

AUTO = tf.data.AUTOTUNE # Chooses optimal number of threads automatically depending on hardware


# %%
# ======== Reading in data + setting parameters =======

DATA_DIR = os.path.join(os.getcwd(), "data", "data_3D_U-net")  # X_*.npy / Y_*.npy

probe = np.load(os.path.join(DATA_DIR, "X_train.npy"), mmap_mode="r")
print("Probe shape:", probe.shape)  # should be (B, 5, 192, 240, 1)

# Sanity check for shape
if probe.ndim != 5:
    raise RuntimeError(f"Expected 5D-Array, have {probe.ndim}Dimension: {probe.shape}")
if probe.shape[-1] != 1:
    raise RuntimeError(f"Expected channel C=1, instead {probe.shape[-1]} in {probe.shape}")
INPUT_SHAPE = tuple(probe.shape[1:])  # -> (5, 192, 240, 1)
del probe

# Define parameters
BATCH_SIZE = 4
EPOCHS     = 100

# check shape for three times (1,2,2)-pooling
D, H, W, C = INPUT_SHAPE
if (H % 8) or (W % 8):
    print(f"[WARN] H={H} or W={W} not divisible by 8; doesn't fit with 3x (1,2,2)-Pooling")

    
def load_split(split):
    """
    Loads data directly from Mmemap
    """
    x = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA_DIR, f"Y_{split}.npy"), mmap_mode="r")
    # Sanity check for shape
    if x.shape[1:] != INPUT_SHAPE or y.shape[1:] != INPUT_SHAPE:
        raise RuntimeError(f"{split}: Datei-Shape {x.shape[1:]} passt nicht zu INPUT_SHAPE {INPUT_SHAPE}")
    return x, y

X_train, Y_train = load_split("train")
X_val,   Y_val   = load_split("val")
X_test,  Y_test  = load_split("test")



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
    u3 = layers.Conv3DTranspose(base_filters*4, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(bn)
    u3 = layers.concatenate([u3, c3])
    c4 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = conv_block(u1, base_filters)

    outputs = layers.Conv3D(1, (1,1,1), activation="linear")(c6)
    return models.Model(inputs, outputs, name="3D_U-Net")



# %%
# =========== Defining Loss function MAE + MS-SSIM (slice-wise) ========

ALPHA = 0.7  # Weight for MS-SSIM

def _flatten_depth(x):
    """
    Making for every depth slice a 2D-image and then evaluate alls slices
    (B,D,H,W,C) -> (B*D, H, W, C)
    """
    shape = tf.shape(x)
    b, d = shape[0], shape[1]
    h, w, c = x.shape[2], x.shape[3], x.shape[4]
    return tf.reshape(x, (b*d, h, w, c))

def ms_ssim_loss(y_true, y_pred, max_val=1.0):
    """
    Defining MS-SSIM for the loss function equivalently as in the paper
    Ensures that values sligthly out of [0,1] are clipped
    """
    y_true_2d = _flatten_depth(tf.clip_by_value(y_true, 0.0, 1.0))
    y_pred_2d = _flatten_depth(tf.clip_by_value(y_pred, 0.0, 1.0))
    ms = tf.image.ssim_multiscale(y_true_2d, y_pred_2d, max_val=max_val)
    return 1.0 - tf.reduce_mean(ms)

def ms_ssim_metric(y_true, y_pred):
    """
    Showing MS-SSIM metric during training
    """
    y_true_2d = _flatten_depth(tf.clip_by_value(y_true, 0.0, 1.0))
    y_pred_2d = _flatten_depth(tf.clip_by_value(y_pred, 0.0, 1.0))
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true_2d, y_pred_2d, max_val=1.0))

def combined_loss(y_true, y_pred):
    """
    Combining the loss composite of MAE and MS-SSIM
    MAE stable and useful for strong signals --> Bragg peaks
    MS-SSIM focuses on structure --> CDW satellite signals
    """
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss(y_true, y_pred, max_val=1.0)  # Data is normalized to [0,1]
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms



# %%
# ======== tf.data input pipeline =======

def make_ds(X_mm, Y_mm, shuffle=True):
    """
    Build a performant tf.data.Dataset pipeline:
    1. Start from indices [0,...,N-1]
    2. Shuffle indices
    3. Map indices -> actual samples via tf_fetch()
    4. Batch samples together
    5. Prefetch to overlap CPU/GPU work
    Result: Dataset yielding batches of (x,y) with shape (B,D,H,W,C)
    """
    n = X_mm.shape[0]
    idx = np.arange(n, dtype=np.int64)

    def _fetch(i):
        """
        Load one sample (X[i], Y[i]) directly from the Numpy memmap on disky
        Input:  index i
        Output: (D,H,W,C) arrays for x and y
        """
        i = int(i)
        return X_mm[i], Y_mm[i]  # (D,H,W,1), float32

    def tf_fetch(i):
        """
        Ensures data is returned as Tensors with fixed shape INPUT_SHAPE.
        Input:  index i
        Output: (x,y) tensors of shape (D,H,W,C)
        """
        x, y = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        x.set_shape(INPUT_SHAPE)
        y.set_shape(INPUT_SHAPE)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle == True:
        ds = ds.shuffle(min(8000, n), reshuffle_each_iteration=True) # Shuffles the data to randomize batches (8000 is buffer size)
    ds = ds.map(tf_fetch, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

train_ds = make_ds(X_train, Y_train, shuffle=True)
val_ds   = make_ds(X_val,   Y_val,   shuffle=False) # Not shuffling the validation data for reproducable results
test_ds  = make_ds(X_test,  Y_test,  shuffle=False) # Not shuffling the test data for reproducable results


# %%
# ======== Compile model =======

model = unet3d(input_shape=INPUT_SHAPE, base_filters=32)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=combined_loss, metrics=["mae", ms_ssim_metric])
model.summary()


# %%
# ======== Callbacks =======

ckpt_dir = os.path.join(os.getcwd(), "checkpoints_3d_unet")
os.makedirs(ckpt_dir, exist_ok=True)

cbs = [
    callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best.keras_V2"),
                              monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]


# %%
# ======== Train =======
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1  # 1:progress bar
)

