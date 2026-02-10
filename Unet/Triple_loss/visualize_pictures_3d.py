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
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Keras")
PIC_DIR        = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Figures")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

# Index der mittleren Slice im Original-Stream (wird zur Mitte des 3D-Stacks)
PICTURE_INDEX = 469

# Modelle
SELECT_LIST = [
    "unet_3d_simple_loss0.0131_val0.0144_epochs100.keras",
    "unet_3d_simple_SSIM_loss0.0608_val0.0794_epochs100.keras",
    "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-084215_loss0.0496_val0.0531.keras",
    "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006_loss0.0489_val0.0524.keras",
    "unet_3d_SSIM_middle__seed42__bf64__D3__lossMAE_SSIM__20251112-180318_loss0.0479_val0.0522.keras", # TODO this doesnt work from the shape, fix this
    "unet_3d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-144919_loss0.0479_val0.0517.keras", # TODO this doesnt work from the shape, fix this
]

def infer_output_mode(model):
    """
    Bestimmt, ob das Modell einen 3D-Stack oder nur eine einzelne 2D-Slice ausgibt.
    - "volume":   (B, D>1, H, W, C)
    - "slice5d":  (B, 1, H, W, C)
    - "slice":    (B, H, W, C) oder (B, H, W)
    """
    out_shape = model.output_shape
    if isinstance(out_shape, list):
        out_shape = out_shape[0]

    if len(out_shape) == 5:
        D_out = out_shape[1]
        # Falls die Depth-Ausgabe 1 ist: wir interpretieren das als einzelne Slice
        if D_out == 1:
            return "slice5d"
        else:
            return "volume"
    elif len(out_shape) == 4:
        return "slice"
    else:
        raise ValueError(
            f"Unerwartetes Output-Shape {out_shape}. "
            "Erwartet 4D (B,H,W,C) oder 5D (B,D,H,W,C)."
        )



def choose_model(list_):
    number = int(input(f"Choose your run (1–{len(list_)}): "))
    index = number - 1
    path = CHECKPOINT_DIR / list_[index]
    print(f"Loading model: {path.name}")
    model = keras.models.load_model(str(path), compile=False)
    return model, number

def infer_depth_from_model(model):
    """
    Erwartetes Input-Shape: (None, D, H, W, C) mit channels_last.
    Liest D aus dem Model-Input.
    """
    # Keras kann Liste von Inputs haben; hier nur einer
    in_shape = model.input_shape
    # Falls mehrere Inputs, nimm den ersten
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    # in_shape z.B. (None, D, H, W, 1)
    if len(in_shape) != 5:
        raise ValueError(f"Unerwartetes Input-Shape {in_shape}. Dieses Script erwartet 5D (B,D,H,W,C).")
    D = in_shape[1]
    if D is None:
        raise ValueError(f"Modell-Depth (D) ist None im Input-Shape {in_shape}.")
    return int(D)

def build_stack_for_depth(h5_path: Path, picture_index: int, depth: int):
    """
    Baut einen 3D-Stack der Tiefe 'depth' um picture_index herum.
    Bei Kanten wird per Clip gearbeitet.
    Rueckgabe: low_stack, high_stack, center_slice_index
    """
    with h5py.File(h5_path, "r") as f:
        low_data  = f["/low_count/data"]   # (H, W, N)
        high_data = f["/high_count/data"]
        H, W, Ntot = low_data.shape

        # Radius aus Tiefe: fuer ungerade depth ist r=(depth-1)//2
        r = (depth - 1) // 2
        # Bei geraden Depths nehmen wir ebenfalls r=floor(depth/2) und definieren center=D//2
        offsets = np.arange(-r, depth - r)  # ergibt z.B. D=5 -> [-2,-1,0,1,2], D=3 -> [-1,0,1], D=7 -> [-3..+3]
        idxs = np.clip(picture_index + offsets, 0, Ntot - 1)

        low_stack  = np.stack([np.asarray(low_data[:,  :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)
        high_stack = np.stack([np.asarray(high_data[:, :, i], dtype=np.float32)  for i in idxs], axis=0)   # (D,H,W)

    center_slice = depth // 2  # fuer ungerade der echte Mittelpunkt; fuer gerade der untere der beiden
    return low_stack, high_stack, center_slice

def val_normalize_per_slice(low_stack, high_stack, scale_val=10000.0):
    """
    Val-Normierung exakt wie im Training-Validation-Pfad:
      1) Clip >= 0
      2) slice-weise Division durch Summe ueber (H,W)
      3) feste Skalierung (10000)
      4) Clip auf [0,1]
    """
    low_stack  = np.clip(low_stack,  0, None)
    high_stack = np.clip(high_stack, 0, None)

    sum_x = np.sum(low_stack,  axis=(1, 2), keepdims=True)  + 1e-12
    sum_y = np.sum(high_stack, axis=(1, 2), keepdims=True) + 1e-12
    x = low_stack  / sum_x
    y = high_stack / sum_y

    x *= scale_val
    y *= scale_val

    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    return x, y

def simple_normalize(image: np.ndarray) -> Normalize:
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    return Normalize(vmin=vmin, vmax=vmax)

# ==== Ablauf ====
model, run_number = choose_model(SELECT_LIST)

D = infer_depth_from_model(model)
mode = infer_output_mode(model)
print(f"Inferierte Depth (D): {D}")
print(f"Inferierter Output-Modus: {mode}")

# Stack aus HDF5 passend zu D bauen
h5_path = DATA_DIR / H5_NAME
low_stack, high_stack, CENTER_SLICE = build_stack_for_depth(h5_path, PICTURE_INDEX, D)

# Val-Normierung wie im Training
x, y = val_normalize_per_slice(low_stack, high_stack, scale_val=10000.0)

# In Tensorform fuer 3D-UNet: (1, D, H, W, 1)
x_norm = tf.convert_to_tensor(x[:, :, :, None][None, ...],  dtype=tf.float32)  # (1,D,H,W,1)
y_norm = tf.convert_to_tensor(y[:, :, :, None][None, ...], dtype=tf.float32)  # (1,D,H,W,1)

# Prediction
y_pred_full = model.predict(x_norm, verbose=0)

if mode == "volume":
    # Ausgabe ist (1, D_out, H, W, 1) mit D_out > 1
    if y_pred_full.ndim != 5:
        raise ValueError(f"Erwartete 5D-Ausgabe fuer 'volume', bekam {y_pred_full.shape}")
    y_pred_center_norm = y_pred_full[0, CENTER_SLICE, :, :, 0]   # (H,W)

elif mode == "slice5d":
    # Ausgabe ist (1, 1, H, W, 1): einzelne Slice mit Dummy-Depth
    if y_pred_full.ndim != 5 or y_pred_full.shape[1] != 1:
        raise ValueError(f"Erwartete (1,1,H,W,C) fuer 'slice5d', bekam {y_pred_full.shape}")
    y_pred_center_norm = y_pred_full[0, 0, :, :, 0]   # (H,W)

elif mode == "slice":
    # Ausgabe ist (1, H, W, 1) oder (1, H, W)
    if y_pred_full.ndim == 4:
        # (1, H, W, C)
        if y_pred_full.shape[-1] >= 1:
            y_pred_center_norm = y_pred_full[0, :, :, 0]
        else:
            raise ValueError(f"Unerwartete Channel-Zahl in 'slice': {y_pred_full.shape}")
    elif y_pred_full.ndim == 3:
        # (1, H, W)
        y_pred_center_norm = y_pred_full[0, :, :]
    else:
        raise ValueError(f"Unerwartete Ausgabe fuer 'slice': {y_pred_full.shape}")
else:
    raise RuntimeError(f"Unbekannter Modus: {mode}")

# Denormierung fuer die mittlere Slice
high_center_slice  = high_stack[CENTER_SLICE]   # (H,W)
y_center_norm_gt   = y[CENTER_SLICE]            # normierte GT-Slice (H,W)

sum_label_center   = float(np.sum(high_center_slice))
sum_y_norm_center  = float(np.sum(y_center_norm_gt))

denorm_factor = max(sum_label_center, 1e-12) / max(sum_y_norm_center, 1e-12)
y_pred_denorm = y_pred_center_norm * denorm_factor

# Für die Darstellung auch die Center-Slices der Inputs
low_img_center  = low_stack[CENTER_SLICE]
high_img_center = high_stack[CENTER_SLICE]





# ==== Visualisierung ====
selected = Path(SELECT_LIST[run_number - 1]).stem
out_png = PIC_DIR / f"{selected}.png"

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), constrained_layout=True, dpi=200)

norm_low  = simple_normalize(low_img_center)
norm_high = simple_normalize(high_img_center)
norm_pred = simple_normalize(y_pred_denorm)

axes[0].imshow(low_img_center,  cmap="gray_r", norm=norm_low,  origin="lower")
axes[1].imshow(high_img_center, cmap="gray_r", norm=norm_high, origin="lower")
axes[2].imshow(y_pred_denorm,   cmap="gray_r", norm=norm_pred, origin="lower")

for ax, title in zip(axes, ["Low (center slice)", "High (center slice)", f"Prediction (center slice) [{mode}]"]):
    ax.set_title(title, fontsize=12)
    ax.axis("off")


PIC_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved visualization to: {out_png}")
