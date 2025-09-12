# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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
# Pfade
DATA_DIR   = os.path.join(os.getcwd(), "data", "data_3D_U-net")  # Ordner mit X_*.npy / Y_*.npy
INPUT_SHAPE = (5, 192, 240, 1)   # NDHWC
BATCH_SIZE  = 4
EPOCHS      = 100
AUTO        = tf.data.AUTOTUNE

# GPU Memory Growth (optional)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

# NPY memmap laden (keine Kopie in RAM)
def load_split(split):
    x = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA_DIR, f"Y_{split}.npy"), mmap_mode="r")
    assert x.shape[1:] == INPUT_SHAPE and y.shape[1:] == INPUT_SHAPE
    return x, y

X_train, Y_train = load_split("train")
X_val,   Y_val   = load_split("val")
X_test,  Y_test  = load_split("test")

# tf.data Pipeline (Index -> numpy_function -> Batch)
def make_ds(X_mm, Y_mm, shuffle=True):
    n = X_mm.shape[0]
    idx = np.arange(n)

    def _fetch(i):
        i = int(i)
        return X_mm[i], Y_mm[i]  # je (5,192,240,1) float32

    def tf_fetch(i):
        x, y = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        x.set_shape(INPUT_SHAPE); y.set_shape(INPUT_SHAPE)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle: ds = ds.shuffle(min(8000, n), reshuffle_each_iteration=True)
    ds = ds.map(tf_fetch, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

train_ds = make_ds(X_train, Y_train, shuffle=True)
val_ds   = make_ds(X_val,   Y_val,   shuffle=False)
test_ds  = make_ds(X_test,  Y_test,  shuffle=False)

# === dein Modell wiederverwenden ===
model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# Callbacks
ckpt_dir = os.path.join(os.getcwd(), "checkpoints_3d_unet"); os.makedirs(ckpt_dir, exist_ok=True)
cbs = [
    callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best.keras"),
                              monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=1)

# Test
res = model.evaluate(test_ds, verbose=1)
print(dict(zip(model.metrics_names, res)))

# Kurzcheck Prediction-Shape
for xb, yb in test_ds.take(1):
    yhat = model.predict(xb, verbose=0)
    print("xb:", xb.shape, "yb:", yb.shape, "yhat:", yhat.shape)

# %%
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf

# Hilfsfunktion, um Vorhersagen & Targets als Numpy-Arrays zu holen
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

# Vorhersagen einsammeln
Y_true, Y_pred = collect_preds_and_targets(model, test_ds)

# Flatten -> 1D-Vektoren (für Metriken wie R² oder SSIM pro Pixel)
yt = Y_true.ravel()
yp = Y_pred.ravel()

# 1) Klassische Fehler-Metriken
mse  = np.mean((yt - yp) ** 2)
mae  = np.mean(np.abs(yt - yp))
rmse = np.sqrt(mse)

# 2) R² Score
r2 = r2_score(yt, yp)

# 3) PSNR (Peak Signal-to-Noise Ratio)
psnr = tf.image.psnr(Y_true, Y_pred, max_val=1.0).numpy().mean()

# 4) SSIM (Structural Similarity Index)
# Achtung: SSIM ist für 2D-Bilder – hier wenden wir es sliceweise an
Y_true_2d = Y_true[:, 2, :, :, :]   # mittlerer Slice aus dem 5er-Block
Y_pred_2d = Y_pred[:, 2, :, :, :]
ssim = tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=1.0).numpy().mean()

print("=== Evaluation on Test Set ===")
print(f"MSE   : {mse:.6f}")
print(f"MAE   : {mae:.6f}")
print(f"RMSE  : {rmse:.6f}")
print(f"R²    : {r2:.6f}")
print(f"PSNR  : {psnr:.2f} dB")
print(f"SSIM  : {ssim:.4f}")

