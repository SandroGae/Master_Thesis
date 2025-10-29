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
ROOT = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jens_stuff import SumScaleNormalizer, compute_denorm_factor_from_xnorm

# Pfade
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\MAE_only_2d")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

PICTURE_INDEX = 469

# Choosing keras models
SELECT_LIST = [
    "unet3d_scout_sweep1_d3_bf8_outsigmoid_lr0_0003_bs32_V1_valloss_3.195e-02_PSNR_23.3.keras",
    "unet3d_scout_sweep2_d3_bf8_outsigmoid_lr3e-05_bs32_V1_valloss_4.308e-02_PSNR_21.3.keras",
    "unet3d_scout_sweep3_d3_bf16_outsigmoid_lr0_0003_bs32_V1_valloss_2.915e-02_PSNR_24.1.keras",    # 3
    "unet3d_scout_sweep4_d3_bf16_outsigmoid_lr3e-05_bs32_V1_valloss_3.956e-02_PSNR_21.7.keras",
    "unet3d_scout_sweep5_d4_bf8_outsigmoid_lr0_0003_bs32_V1_valloss_2.667e-02_PSNR_24.keras",
    "unet3d_scout_sweep6_d4_bf8_outsigmoid_lr3e-05_bs32_V1_valloss_4.164e-02_PSNR_21.4.keras",      # 6
    "unet3d_scout_sweep7_d4_bf16_outsigmoid_lr0_0003_bs32_V1_valloss_2.880e-02_PSNR_24.keras",
    "unet3d_scout_sweep8_d4_bf16_outsigmoid_lr3e-05_bs32_V1_valloss_3.737e-02_PSNR_22.keras",
    "unet3d_scout_sweep9_d4_bf24_outsigmoid_lr0_0003_bs32_V1_valloss_2.947e-02_PSNR_23.8.keras",    # 9
    "unet3d_scout_sweep10_d4_bf24_outsigmoid_lr1e-05_bs32_V1_valloss_4.107e-02_PSNR_21.2.keras",
    "unet3d_scout_sweep11_d4_bf32_outsigmoid_lr0_0003_bs32_V1_valloss_2.895e-02_PSNR_24.keras",
    "unet3d_scout_sweep12_d4_bf32_outsigmoid_lr1e-05_bs32_V1_valloss_3.824e-02_PSNR_21.9.keras",     # 12
    "unet3d_scout_sweep13_d4_bf32_outsigmoid_lr0_0003_bs32_V1_valloss_2.488e-02_PSNR_24.5.keras",    # 100 epochs, my unet
    "unet3d_scout_sweep14_d4_bf68_outsigmoid_lr0_0003_bs32_V1_valloss_2.827e-02_PSNR_24.3.keras"     # 100 epochs, VDSR Jens
]


def choose_model(list):
    """Choose the model you want to load"""
    number = int(input("Choose your run (1–N): "))
    index = number - 1
    path = CHECKPOINT_DIR / list[index]
    print(f"Loading model: {path.name}")
    model = load_model(str(path), compile=False)
    return model, number


model, run_number = choose_model(SELECT_LIST)

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
    scale_min=5000,
    scale_max=15000,
    pre_offset=0.0,
    normalize_label=True,
    batch_mode=True, # Jedes Sample separat normalisieren
)
x = tf.convert_to_tensor(low_img[None, None, :, :, None], dtype=tf.float32)  # (1,1,H,W,1)
y = tf.convert_to_tensor(high_img[None, None, :, :, None], dtype=tf.float32) # (1,1,H,W,1)
x_norm, y_norm = normalizer.map(x, y)

# Prediction
y_pred = model.predict(x_norm, verbose=0)[0, 0, :, :, 0]   # (H,W)

# Denormaliesierung
sum_label  = tf.reduce_sum(y)                      # Summe High in Originaleinheiten
sum_y_norm = tf.reduce_sum(y_norm)                 # Summe normalisiertes Label
denorm_factor = tf.maximum(sum_label, 1e-12) / tf.maximum(sum_y_norm, 1e-12)
y_pred_denorm = y_pred * float(denorm_factor.numpy())


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
save_dir = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Keras_Models\MAE_only_2d")
out_png = save_dir / f"visualized_sweep_{run_number}.png"

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