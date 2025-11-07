# visualize_pictures_2d.py
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
PICTURE_INDEX  = 469

# Auswahl der Modelle
SELECT_LIST = [
    "VDSR_reconstruction_V2_loss0.0132_val0.0133_epochs100.keras",
    "unet_2d_simple_loss0.0135_val0.0142_epochs100.keras",
    "unet_2d_simple_loss0.0118_val0.0135_epochs200.keras",
    "unet_2d_SSIM_loss0.0605_val0.0787_epochs200.keras"
]

def choose_model(list_):
    """Choose the model you want to load"""
    number = int(input("Choose your run (1–N): "))
    index = number - 1
    path = CHECKPOINT_DIR / list_[index]
    print(f"Loading model: {path.name}")
    model = keras.models.load_model(str(path), compile=False)
    return model, number

model, run_number = choose_model(SELECT_LIST)

# Lade Testdaten
h5_path = DATA_DIR / H5_NAME
with h5py.File(h5_path, "r") as f:
    low_data  = f["/low_count/data"]   # (H, W, N)
    high_data = f["/high_count/data"]
    low_img  = np.asarray(low_data[:, :, PICTURE_INDEX], dtype=np.float32)
    high_img = np.asarray(high_data[:, :, PICTURE_INDEX], dtype=np.float32)

# === Normalisierung exakt wie im Validation-Datensatz des Trainings ===
# 1) Clipping auf [0, ∞)
low_img  = np.clip(low_img,  0, None)
high_img = np.clip(high_img, 0, None)

# 2) Normierung pro Bild durch Summe
sum_low  = np.sum(low_img)  + 1e-12
sum_high = np.sum(high_img) + 1e-12
x = low_img  / sum_low
y = high_img / sum_high

# 3) Feste Skalierung in [10000, 10001] (Val-Modus)
scale = 10000.0
x *= scale
y *= scale

# 4) Clipping auf [0, 1]
x = np.clip(x, 0, 1)
y = np.clip(y, 0, 1)

# 5) In Tensorform (immer 4D, da 2D-Modelle)
x_norm = tf.convert_to_tensor(x[None, :, :, None], dtype=tf.float32)  # (1,H,W,1)
y_norm = tf.convert_to_tensor(y[None, :, :, None], dtype=tf.float32)

# Prediction (immer 2D-Modell!)
y_pred = model.predict(x_norm, verbose=0)[0, :, :, 0]

# Denormalisierung wie bisher
sum_label  = np.sum(high_img)
sum_y_norm = np.sum(y)
denorm_factor = np.maximum(sum_label, 1e-12) / np.maximum(sum_y_norm, 1e-12)
y_pred_denorm = y_pred * denorm_factor

# === Visualisierung ===
def simple_normalize(image: np.ndarray) -> Normalize:
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    return Normalize(vmin=vmin, vmax=vmax)

selected = Path(SELECT_LIST[run_number - 1]).stem
out_png = PIC_DIR / f"{selected}.png"

fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True, dpi=200)

norm_low  = simple_normalize(low_img)
norm_high = simple_normalize(high_img)
norm_pred = simple_normalize(y_pred_denorm)

axes[0].imshow(low_img,       cmap="gray_r", norm=norm_low,  origin="lower")
axes[1].imshow(high_img,      cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm, cmap="gray_r", norm=norm_pred, origin="lower")

for ax, title in zip(axes, ["Low", "High", "Prediction"]):
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved visualization to: {out_png}")
