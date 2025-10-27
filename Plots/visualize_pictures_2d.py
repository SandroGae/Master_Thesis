# visualize_pictures_2d.py
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Projekt-Root in sys.path (damit jens_stuff auffindbar ist)
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jens_stuff import SumScaleNormalizer

# Pfade
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

PICTURE_INDEX = 469

# Choosing keras models
SELECT_LIST = [
    "unet_3d_JENS_V2_sweep2_d3_bf8_ReLU_LN_outsigmoid_lr3e-05_bs32__mae_V1_valloss_8.804e-02_PSNR_27.2.keras"
]


def choose_model(list):
    """Choose the model you want to load"""
    number = int(input("Choose your run: "))
    index = number - 1
    path = CHECKPOINT_DIR / list[index]
    print(type(path))
    return load_model(str(path), compile=False)


model = choose_model(SELECT_LIST)

# Lade Testdaten
h5_path = DATA_DIR / H5_NAME
with h5py.File(h5_path, "r") as f:
    low_data  = f["/low_count/data"]   # (H, W, N)
    high_data = f["/high_count/data"]
    # gewünschtes Bild laden
    low_img  = np.asarray(low_data[:, :, PICTURE_INDEX], dtype=np.float32)
    high_img = np.asarray(high_data[:, :, PICTURE_INDEX], dtype=np.float32)


# Normalisierung Jens Stil
normalizer = SumScaleNormalizer(
    scale_range=[5000, 15001],
    pre_offset=0.0,
    normalize_label=True,
    axis=None,
    batch_mode=True, # Jedes Sample separat normalisieren
    clip_before=[0., float("inf")],
    clip_after=[0., 1.]
)
x = tf.convert_to_tensor(low_img[None, None, :, :, None], dtype=tf.float32)  # (1,1,H,W,1)
y = tf.convert_to_tensor(high_img[None, None, :, :, None], dtype=tf.float32) # (1,1,H,W,1)
x_norm, y_norm = normalizer.map(x, y)

# Prediction
y_pred = model.predict(x_norm, verbose=0)[0, 0, :, :, 0]   # (H,W)

# --- Denorm: benutze LABEL-Summe, nicht feature-sum ---
scale_val  = float(tf.reshape(normalizer._denorm_pars['scale'], ()).numpy())  # skalar
sum_label  = float(np.maximum(np.sum(high_img), 1e-12))                        # skalar

y_pred_denorm = y_pred * (sum_label / scale_val)



# Visualisierung
def simple_normalize(image: np.ndarray) -> Normalize:
    """
    Kleinste 0.1%-Werte weiss, grösste 99.9%-Werte schwarz.
    """
    # Berechne die Perzentile 0.1 und 99.9
    vmin, vmax = np.percentile(image, [0.5, 99.5])

    # Falls die Werte unbrauchbar sind (NaN, unendlich oder gleich)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(image))
        vmax = float(np.nanmax(image) + 1e-6)

    return Normalize(vmin=vmin, vmax=vmax)


# Zielpfad
save_dir = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\plots_serie_11_frame_18")
out_png = save_dir / "viz_p1p99_serie11_frame18.png"

# Figuren-Setup
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), constrained_layout=True, dpi=200)

# Normierungen pro Bild
norm_low  = simple_normalize(low_img)
norm_high = simple_normalize(high_img)
norm_pred = simple_normalize(y_pred_denorm)

# Plotten (0=weiss, max=schwarz via gray_r)
axes[0].imshow(low_img,        cmap="gray_r", norm=norm_low,  origin="lower")
axes[1].imshow(high_img,       cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm,  cmap="gray_r", norm=norm_pred, origin="lower")

# Titel & Speichern
titles = ["Low", "High", "Prediction"]
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)