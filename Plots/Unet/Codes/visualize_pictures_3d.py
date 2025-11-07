# visualize_pictures_3d.py
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib.colors import Normalize

# Pfade
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\Unet")
PIC_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\Unet\Figures")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

PICTURE_INDEX = 469   # Mittelpunkt der 5er-Scheibe
STACK_RADIUS  = 2     # ergibt Tiefe D=5
CENTER_SLICE  = 2     # mittlere Slice in [0..4]

# Modelle
SELECT_LIST = [
    "unet_3d_simple_loss0.0131_val0.0144_epochs100.keras",
    "unet_3d_simple_SSIM_loss0.0608_val0.0794_epochs100.keras"
]

def choose_model(list_):
    number = int(input("Choose your run (1–N): "))
    index = number - 1
    path = CHECKPOINT_DIR / list_[index]
    print(f"Loading model: {path.name}")
    model = keras.models.load_model(str(path), compile=False)
    return model, number

model, run_number = choose_model(SELECT_LIST)

# --- Daten laden & 5er-Stack bauen ---
h5_path = DATA_DIR / H5_NAME
with h5py.File(h5_path, "r") as f:
    low_data  = f["/low_count/data"]   # (H, W, N)
    high_data = f["/high_count/data"]

    H, W, Ntot = low_data.shape
    offsets = np.arange(-STACK_RADIUS, STACK_RADIUS + 1)  # [-2,-1,0,1,2]
    idxs = np.clip(PICTURE_INDEX + offsets, 0, Ntot - 1)

    low_stack  = np.stack([np.asarray(low_data[:,  :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)
    high_stack = np.stack([np.asarray(high_data[:, :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)

# --- Val-Normalisierung (slice-weise) ---
# 1) Clipping auf [0, ∞)
low_stack  = np.clip(low_stack,  0, None)
high_stack = np.clip(high_stack, 0, None)

# 2) Normierung pro Slice durch Summe über (H,W)
sum_x = np.sum(low_stack,  axis=(1, 2), keepdims=True)  + 1e-12  # (D,1,1)
sum_y = np.sum(high_stack, axis=(1, 2), keepdims=True) + 1e-12
x = low_stack  / sum_x
y = high_stack / sum_y

# 3) Feste Skalierung (Val-Modus)
scale = 10000.0
x *= scale
y *= scale

# 4) Clipping auf [0, 1]
x = np.clip(x, 0, 1)
y = np.clip(y, 0, 1)

# 5) In Tensorform für 3D-UNet: (1, D, H, W, 1)
x_norm = tf.convert_to_tensor(x[:, :, :, None][None, ...],  dtype=tf.float32)  # (1,5,H,W,1)
y_norm = tf.convert_to_tensor(y[:, :, :, None][None, ...], dtype=tf.float32)  # (1,5,H,W,1)

# --- Prediction (ohne Fallback, das Modell ist 3D und erwartet 5D) ---
y_pred_full = model.predict(x_norm, verbose=0)               # (1,5,H,W,1)
y_pred_center = y_pred_full[0, CENTER_SLICE, :, :, 0]        # (H,W)

# --- Denormalisierung für die mittlere Slice ---
sum_label_center  = float(np.sum(high_stack[CENTER_SLICE]))  # Summe im originalen High (Slice)
sum_y_norm_center = float(np.sum(y[CENTER_SLICE]))           # Summe im normalisierten High (Slice)
denorm_factor = max(sum_label_center, 1e-12) / max(sum_y_norm_center, 1e-12)
y_pred_denorm = y_pred_center * denorm_factor

# Für die Darstellung auch die Center-Slices der Inputs
low_img_center  = low_stack[CENTER_SLICE]
high_img_center = high_stack[CENTER_SLICE]

# --- Visualisierung ---
def simple_normalize(image: np.ndarray) -> Normalize:
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    return Normalize(vmin=vmin, vmax=vmax)

selected = Path(SELECT_LIST[run_number - 1]).stem
out_png = PIC_DIR / f"{selected}.png"

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), constrained_layout=True, dpi=200)

norm_low  = simple_normalize(low_img_center)
norm_high = simple_normalize(high_img_center)
norm_pred = simple_normalize(y_pred_denorm)

axes[0].imshow(low_img_center,  cmap="gray_r", norm=norm_low,  origin="lower")
axes[1].imshow(high_img_center, cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm,   cmap="gray_r", norm=norm_pred, origin="lower")

for ax, title in zip(axes, ["Low (center slice)", "High (center slice)", "Prediction (center slice)"]):
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved visualization to: {out_png}")
