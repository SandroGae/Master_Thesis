# visualize_pictures_3d.py
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow import keras
load_model = keras.models.load_model
from matplotlib.colors import Normalize

# Projekt-Root in sys.path (damit jens_stuff auffindbar ist)
ROOT = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jens_stuff import SumScaleNormalizer

# Pfade
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\MAE_only")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

PICTURE_INDEX = 469  # Mittelpunkt der 5er-Scheibe
STACK_RADIUS  = 2    # ergibt Tiefe D=5
CENTER_SLICE  = 2    # mittlere Slice in [0..4]

# Choosing keras models
SELECT_LIST = [
    "unet_3d_simple_loss0.0131_val0.0144_epochs100.keras"
]

def choose_model(list_):
    number = int(input("Choose your run (1–N): "))
    index = number - 1
    path = CHECKPOINT_DIR / list_[index]
    print(f"Loading model: {path.name}")
    model = load_model(str(path), compile=False)
    return model, number

model, run_number = choose_model(SELECT_LIST)

# Lade Testdaten
h5_path = DATA_DIR / H5_NAME
with h5py.File(h5_path, "r") as f:
    low_data  = f["/low_count/data"]   # (H, W, N)
    high_data = f["/high_count/data"]

    H, W, Ntot = low_data.shape

    # 5er-Indices um PICTURE_INDEX, an den Rändern geclamped
    offsets = np.arange(-STACK_RADIUS, STACK_RADIUS + 1)  # [-2,-1,0,1,2]
    idxs = np.clip(PICTURE_INDEX + offsets, 0, Ntot - 1)

    # 3D-Stacks (D,H,W) -> wir bauen direkt (D,H,W,1)
    low_stack  = np.stack([np.asarray(low_data[:,  :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)
    high_stack = np.stack([np.asarray(high_data[:, :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)

# Normalisierung Jens Stil (Batch-Modus; pro Sample/Slice)
normalizer = SumScaleNormalizer(
    scale_min=10000,
    scale_max=10001,
    pre_offset=0.0,
    normalize_label=True,
    batch_mode=True,
)

# Eingabeformen:
# x: (1, D, H, W, 1), y: (1, D, H, W, 1)
x = tf.convert_to_tensor(low_stack[:,  :, :,  None][None, ...],  dtype=tf.float32)  # (1,5,H,W,1)
y = tf.convert_to_tensor(high_stack[:, :, :,  None][None, ...], dtype=tf.float32)   # (1,5,H,W,1)

x_norm, y_norm = normalizer.map(x, y)

# Prediction
try:
    # 3D-Modell erwartet 5D: (B,D,H,W,C) -> (B,D,H,W,C)
    y_pred_full = model.predict(x_norm, verbose=0)  # (1,5,H,W,1)
    y_pred_center = y_pred_full[0, CENTER_SLICE, :, :, 0]  # (H,W)
except Exception:
    # Fallback: Modell ist 2D und erwartet 4D; nimm mittlere Slice
    x4 = x_norm[:, CENTER_SLICE, ...]  # (1,H,W,1)
    y_pred_center = model.predict(x4, verbose=0)[0, :, :, 0]  # (H,W)

# Denormalisierung SLICE-spezifisch (mittlere Slice)
# Summe im originalen High (Slice) vs. normalisiertem High (Slice)
sum_label_center  = tf.reduce_sum(y[0, CENTER_SLICE, :, :, 0])       # scalar
sum_y_norm_center = tf.reduce_sum(y_norm[0, CENTER_SLICE, :, :, 0])  # scalar
denorm_factor = tf.maximum(sum_label_center, 1e-12) / tf.maximum(sum_y_norm_center, 1e-12)
y_pred_denorm = y_pred_center * float(denorm_factor.numpy())

# Auch die entsprechenden Low/High-Slices fuer die Visualisierung
low_img_center  = low_stack[CENTER_SLICE,  :, :]
high_img_center = high_stack[CENTER_SLICE, :, :]

# Visualisierung
def simple_normalize(image: np.ndarray) -> Normalize:
    """
    Kleinste 0.5%-Werte weiss, groesste 99.5%-Werte schwarz.
    """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    return Normalize(vmin=vmin, vmax=vmax)

# Zielpfad
save_dir = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\MAE_only")
out_png = save_dir / f"visualized_sweep_{run_number}_3Dcenter.png"

# Figuren-Setup
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), constrained_layout=True, dpi=200)

# Normierungen pro Bild
norm_low  = simple_normalize(low_img_center)
norm_high = simple_normalize(high_img_center)
norm_pred = simple_normalize(y_pred_denorm)

# Plotten (0=weiss, max=schwarz via gray_r)
axes[0].imshow(low_img_center,       cmap="gray_r", norm=norm_low,  origin="lower")
axes[1].imshow(high_img_center,      cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm,        cmap="gray_r", norm=norm_pred, origin="lower")

# Titel & Speichern
titles = ["Low (center slice)", "High (center slice)", "Prediction (center slice)"]
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)
