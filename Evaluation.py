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
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score


# %%
# ======== Load Test Data ========

DATA_DIR = os.path.join(os.getcwd(), "data", "data_3D_U-net")

# Shape check from one file
probe = np.load(os.path.join(DATA_DIR, "X_test.npy"), mmap_mode="r")
INPUT_SHAPE = tuple(probe.shape[1:])  # (D,H,W,C)
print("Detected INPUT_SHAPE:", INPUT_SHAPE)
del probe

BATCH_SIZE = 4
AUTO = tf.data.AUTOTUNE

def load_split(split):
    x = np.load(os.path.join(DATA_DIR, f"X_{split}.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA_DIR, f"Y_{split}.npy"), mmap_mode="r")
    if x.shape[1:] != INPUT_SHAPE or y.shape[1:] != INPUT_SHAPE:
        raise RuntimeError(f"{split} shape mismatch: {x.shape[1:]} vs {INPUT_SHAPE}")
    return x, y

X_test, Y_test = load_split("test")

def make_ds(X_mm, Y_mm, shuffle=False):
    n = X_mm.shape[0]
    idx = np.arange(n, dtype=np.int64)

    def _fetch(i):
        i = int(i)
        return X_mm[i], Y_mm[i]

    def tf_fetch(i):
        x, y = tf.numpy_function(_fetch, [i], [tf.float32, tf.float32])
        x.set_shape(INPUT_SHAPE)
        y.set_shape(INPUT_SHAPE)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle:
        ds = ds.shuffle(min(8000, n), reshuffle_each_iteration=True)
    ds = ds.map(tf_fetch, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

# Build dataset for evaluation
test_ds = make_ds(X_test, Y_test, shuffle=False)

# %%
# ======== Load best saved model & evaluate on test ========

# Open Model that should be evaluated
ckpt_dir = os.path.join(os.getcwd(), "checkpoints_3d_unet")
best_path = os.path.join(ckpt_dir, "best.keras") # Pick model to evaluate

print(f"Load Model: {best_path}")
best_model = tf.keras.models.load_model(best_path, custom_objects={
        "combined_loss": globals().get("combined_loss"),
        "ms_ssim_loss": globals().get("ms_ssim_loss"),
        "ms_ssim_metric": globals().get("ms_ssim_metric"),
    }
)

# Collect all predictions and targets from the test dataset
def collect_preds_and_targets(model, dataset, max_batches=None):
    y_true, y_pred = [], []
    for b, (xb, yb) in enumerate(dataset):
        yhat = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(yhat)
        if max_batches and (b + 1) >= max_batches:
            break
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred

# Optional: testing less batches, e.g.. MAX_BATCHES = 50
MAX_BATCHES = None

# Evaluation
Y_true, Y_pred = collect_preds_and_targets(best_model, test_ds, max_batches=MAX_BATCHES)

yt = Y_true.ravel()
yp = Y_pred.ravel()

mse  = np.mean((yt - yp) ** 2)
mae  = np.mean(np.abs(yt - yp))
rmse = np.sqrt(mse)
r2   = r2_score(yt, yp)

# PSNR over 3D-Volume (Data in [0,1])
psnr = tf.image.psnr(Y_true, Y_pred, max_val=1.0).numpy().mean()

# SSIM slice wise (mittlerer Slice entlang D)
Y_true_2d = Y_true[:, Y_true.shape[1] // 2, :, :, :]
Y_pred_2d = Y_pred[:, Y_pred.shape[1] // 2, :, :, :]
ssim = tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=1.0).numpy().mean()

print("=== Evaluation on Test Set (loaded best checkpoint) ===")
print(f"MSE   : {mse:.6f}")
print(f"MAE   : {mae:.6f}")
print(f"RMSE  : {rmse:.6f}")
print(f"R2    : {r2:.6f}")
print(f"PSNR  : {psnr:.2f} dB")
print(f"SSIM  : {ssim:.4f}")

