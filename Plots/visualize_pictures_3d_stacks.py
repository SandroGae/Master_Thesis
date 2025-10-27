# visualize_pictures_2d_stacks.py
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
from train_utils import H5Groups

# Pfade
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

PICTURE_INDEX = 469

# Choosing keras models
SELECT_LIST = [
    "sweep_d3_bf8_ELU_LN_outsigmoid_lr3e-05_bs128__mae_ssim_0_7_V1_valloss_2.497e-01_PSNR_21.4.keras",
    "sweep_d4_bf16_ELU_LN_outsigmoid_lr0_0001_bs32__mae_ssim_0_7_V1_valloss_1.874e-01_PSNR_24.2.keras",
    "unet_3d_JENS_V2_sweep1_d3_bf8_ELU_LN_outsigmoid_lr0_0003_bs32__mae_ssim_0_7_V2_valloss_4.575e-01_PSNR_11.6.keras",
    "unet_3d_JENS_V2_sweep32_d4_bf16_ELU_LN_outsigmoid_lr0_0001_bs128__mae_ssim_0_7_V1_valloss_6.600e-02_PSNR_30.1.keras",
    "unet_3d_JENS_V2_sweep2_d3_bf8_ELU_LN_outsigmoid_lr3e-05_bs32__mae_ssim_0_7_V1_valloss_8.599e-02_PSNR_27.6.keras",          # This one is the 3D!! = 5
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

# Parameter der Gruppenlogik
GROUP_LEN = 41
SERIES_INDEX = 11
VOLUME_INDEX = 16
STACK_DEPTH = 5 
CENTER_SLICE = 2  # Mittleres Bild (0 basiert)

# Reader initialisieren
reader = H5Groups(DATA_DIR / "test_data.hdf5", group_len=GROUP_LEN)
# Komplette Serie (41 Bilder) laden + alle 37 Volumina generieren
low_stacks, high_stacks = reader.get_group_windows(SERIES_INDEX) # Shapes: (37, 5, H, W, 1)
# Gewünschtes Volumen
low_vol  = low_stacks [VOLUME_INDEX:VOLUME_INDEX + 1, ...]   # (1,5,H,W,1)
high_vol = high_stacks[VOLUME_INDEX:VOLUME_INDEX + 1, ...]   # (1,5,H,W,1)

# Mittleren Slice herausnehmen für Visualisierung
low_img_center  = low_vol [0, CENTER_SLICE, ..., 0]
high_img_center = high_vol[0, CENTER_SLICE, ..., 0]



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
x = tf.convert_to_tensor(low_vol,  dtype=tf.float32)  # (1,5,H,W,1)
y = tf.convert_to_tensor(high_vol, dtype=tf.float32)  # (1,5,H,W,1)

x_norm, y_norm = normalizer.map(x, y)

y_pred = model.predict(x_norm, verbose=0)                 # (1,5,H,W,1)
y_pred_c = y_pred[0, CENTER_SLICE, ..., 0]                # (H,W)

# Skalierungsfaktor aus dem Normalizer
scale_factor = float(normalizer._denorm_pars["scale"].numpy())
# Summe aller Pixel des High-Count-Mittelslices
true_sum = float(np.sum(high_vol[0, CENTER_SLICE, ..., 0]))
# Schutz vor Division durch Null
if true_sum < 1e-12:
    true_sum = 1e-12
# Rückskalieren der Prediction in den realen Intensitätsbereich
y_pred_denorm_c = y_pred_c * (true_sum / scale_factor)




# Visualisierung
def simple_normalize(image: np.ndarray):
    """
    Kleinste 0.1%-Werte weiss, grösste 99.9%-Werte schwarz.
    """
    # Berechne die Perzentile
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    # Falls die Werte unbrauchbar sind (NaN, unendlich oder gleich)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(image))
        vmax = float(np.nanmax(image) + 1e-6)

    return Normalize(vmin=vmin, vmax=vmax)


# Zielpfad
save_dir = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\plots_serie_11_frame_18")
out_png = save_dir / "viz_p1p99_serie11_frame18_3D.png"

# Figuren-Setup
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), constrained_layout=True, dpi=200)

# Normierungen pro Bild
norm_low  = simple_normalize(low_img_center)
norm_high = simple_normalize(high_img_center)
norm_pred = simple_normalize(y_pred_denorm_c)

axes[0].imshow(low_img_center, cmap="gray_r", norm=norm_low, origin="lower")
axes[1].imshow(high_img_center, cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm_c, cmap="gray_r", norm=norm_pred, origin="lower")

# Titel & Speichern
titles = ["Low", "High", "Prediction"]
for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=12)
    ax.axis("off")

fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)