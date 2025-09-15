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
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import numpy as np


# %%
def convolutional_block(input_tensor, number_of_filters, kernel_size=(3, 3, 3), padding_mode="same", activation_function="relu"):
    """Two convolutions with BatchNorm"""
    x = layers.Conv3D(filters=number_of_filters, kernel_size=kernel_size, padding=padding_mode)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)

    x = layers.Conv3D(filters=number_of_filters, kernel_size=kernel_size, padding=padding_mode)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)
    return x

def build_unet3d(input_shape=(5, 192, 240, 1), base_number_of_filters=32):
    """3D U-Net for sequences of 2D pictures (Pooling only in height and width)."""
    input_layer = layers.Input(shape=input_shape)

    # Encoder
    encoder_block1 = convolutional_block(input_layer, base_number_of_filters)
    downsampled1   = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(encoder_block1)

    encoder_block2 = convolutional_block(downsampled1, base_number_of_filters * 2)
    downsampled2   = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(encoder_block2)

    encoder_block3 = convolutional_block(downsampled2, base_number_of_filters * 4)
    downsampled3   = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(encoder_block3)

    # Bottleneck
    bottleneck = convolutional_block(downsampled3, base_number_of_filters * 8)

    # Decoder
    upsampled3 = layers.Conv3DTranspose(filters=base_number_of_filters * 4, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(bottleneck)
    merged3    = layers.concatenate([upsampled3, encoder_block3])
    decoder_block3 = convolutional_block(merged3, base_number_of_filters * 4)

    upsampled2 = layers.Conv3DTranspose(filters=base_number_of_filters * 2, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(decoder_block3)
    merged2    = layers.concatenate([upsampled2, encoder_block2])
    decoder_block2 = convolutional_block(merged2, base_number_of_filters * 2)

    upsampled1 = layers.Conv3DTranspose(filters=base_number_of_filters, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="same")(decoder_block2)
    merged1    = layers.concatenate([upsampled1, encoder_block1])
    decoder_block1 = convolutional_block(merged1, base_number_of_filters)

    # Output
    output_layer = layers.Conv3D(filters=1, kernel_size=(1, 1, 1), activation="linear")(decoder_block1)

    return models.Model(inputs=input_layer, outputs=output_layer, name="3D_U-Net")


# %%
# === Loading Data ===

# Define Parameters
INPUT_SHAPE = (5, 192, 240, 1)
BATCH_SIZE  = 4
EPOCHS      = 50
CKPT_DIR    = os.path.join(os.getcwd(), "checkpoints_3d_unet")

DATA_DIR = os.path.join(os.getcwd(), "data", "data_3D_U-net")  # <— Data FOlder
CKPT_DIR  = os.path.join(os.getcwd(), "checkpoints_3d_unet")   # <— best.keras gets saved here
os.makedirs(CKPT_DIR, exist_ok=True)

AUTO = tf.data.AUTOTUNE

# GPU Memory Growth
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

def load_split(split):
    X = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(DATA_DIR, f"Y_{split}.npy"), mmap_mode="r")
    assert X.shape[1:] == INPUT_SHAPE and Y.shape[1:] == INPUT_SHAPE
    return X, Y

def make_ds(X, Y, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    def _fetch(i):
        i = int(i);  return X[i], Y[i]
    def tf_fetch(i):
        x, y = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        x.set_shape(INPUT_SHAPE); y.set_shape(INPUT_SHAPE)
        return x, y
    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle: ds = ds.shuffle(min(8000, n), reshuffle_each_iteration=True)
    return ds.map(tf_fetch, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

X_train, Y_train = load_split("train")
X_val,   Y_val   = load_split("val")
X_test,  Y_test  = load_split("test")

train_ds = make_ds(X_train, Y_train, True)
val_ds   = make_ds(X_val,   Y_val,   False)
test_ds  = make_ds(X_test,  Y_test,  False)


# %%
# === Training the Model ===

# Define Model
model = build_unet3d(input_shape=INPUT_SHAPE, base_number_of_filters=32)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="mse",metrics=["mae"])
model.summary()

# Callbacks
cbs = [
    callbacks.ModelCheckpoint(
        filepath=os.path.join(CKPT_DIR, "best.keras_V2"), # Saving the best model
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Training Model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=1)

# %%
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score


# Loading test data (again to make this snippet independent)
BASE_DIR   = os.getcwd()
DATA_DIR   = os.path.join(BASE_DIR, "data", "data_3D_U-net")
CKPT_PATH  = os.path.join(BASE_DIR, "checkpoints_3d_unet", "best.keras")
BATCH_SIZE = 4
AUTO       = tf.data.AUTOTUNE

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"), mmap_mode="r")
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"), mmap_mode="r")
INPUT_SHAPE = X_test.shape[1:]  # (5,192,240,1)
print("X_test shape:", X_test.shape, "| per-sample shape:", INPUT_SHAPE)
print("Y_test shape:", Y_test.shape)

def make_ds_from_memmap(X_mm, Y_mm, batch_size=4, shuffle=False):
    n = X_mm.shape[0]
    idx = np.arange(n)

    def _fetch(i):
        i = int(i)
        return X_mm[i], Y_mm[i]

    def tf_fetch(i):
        x, y = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        x.set_shape(INPUT_SHAPE); y.set_shape(INPUT_SHAPE)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle: ds = ds.shuffle(min(8000, n), reshuffle_each_iteration=False)
    ds = ds.map(tf_fetch, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size).prefetch(AUTO)
    return ds

test_ds = make_ds_from_memmap(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 2) Modell laden (kein Training)
#    -> optional mit kombinierter Loss/Metric kompilieren
# ----------------------------
model = tf.keras.models.load_model(
    CKPT_PATH,
    compile=False,                 # wir kompilieren gleich neu für die Auswertung
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",                    # Standard: MSE
    metrics=["mae"]                # Standard: MAE
)
print("Loaded model from:", CKPT_PATH)
model.summary()

# ----------------------------
# 3) Vorhersagen sammeln
# ----------------------------
def collect_preds_and_targets(model, dataset, max_batches=None):
    y_true, y_pred = [], []
    for b, (xb, yb) in enumerate(dataset):
        yhat = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(yhat)
        if max_batches and b >= max_batches - 1:
            break
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred

Y_true, Y_pred = collect_preds_and_targets(model, test_ds)

# ----------------------------
# 4) Metriken & kombinierte Loss reporten
# ----------------------------
max_val = 1.0  # falls deine Daten 0..1 skaliert sind

yt = Y_true.ravel()
yp = Y_pred.ravel()
mse  = np.mean((yt - yp) ** 2)
mae  = np.mean(np.abs(yt - yp))
rmse = np.sqrt(mse)
r2   = r2_score(yt, yp)

# 2D-Stack für PSNR/SSIM/MS-SSIM
N, D, H, W, C = Y_true.shape
Y_true_2d = Y_true.reshape(N * D, H, W, C).astype(np.float32)
Y_pred_2d = Y_pred.reshape(N * D, H, W, C).astype(np.float32)

psnr    = tf.image.psnr(Y_true_2d, Y_pred_2d, max_val=max_val).numpy().mean()
ssim    = tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=max_val).numpy().mean()
ms_ssim = tf.image.ssim_multiscale(Y_true_2d, Y_pred_2d, max_val=max_val).numpy().mean()

print("=== Evaluation on Test Set (loaded model) ===")
print(f"MSE       : {mse:.6f}")
print(f"MAE       : {mae:.6f}")
print(f"RMSE      : {rmse:.6f}")
print(f"R²        : {r2:.6f}")
print(f"PSNR      : {psnr:.2f} dB")
print(f"SSIM      : {ssim:.4f}")
print(f"MS-SSIM   : {ms_ssim:.4f}")

