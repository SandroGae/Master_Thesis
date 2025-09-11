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
"""
This dataset could be used to either train a 2.5D CNN that uses the different low count pictures as channels
Or it could be used to train a CNN model with 3D inputs and 2D outputs / 2D ground truth (simply using 3D kernels)
"""

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = "test_data.hdf5"

with h5py.File(file_path, "r") as f:
    high_data = f["/high_count/data"][:]   # (192, 240, 3034)
    low_data = f["/low_count/data"][:]     # (192, 240, 3034)

# Transpose: Matlab --> Python
high_data = high_data.transpose(2, 0, 1)   # (3034, 192, 240)
low_data = low_data.transpose(2, 0, 1)     # (3034, 192, 240)


# %%
def anscombe_vst(x):
    # Negative values get yeeted to zero (counts should not be negative)
    x = np.maximum(x, 0)
    return 2.0 * np.sqrt(x + 3.0/8.0)

def compute_clip_from_high(high_data, percentile=99.9, use_vst=True, max_samples=5_000_000, rng=None):
    """
    From High count data, determine global Clip_values
    percentile: e.g. 99.9
    use_vst: If True -> Calculate Clip on VSC domain (usually better)
    """
    rng = np.random.default_rng() if rng is None else rng
    arr = high_data.ravel()

    if arr.size > max_samples:
        idx = rng.choice(arr.size, size=max_samples, replace=False)
        sample = arr[idx]
    else:
        sample = arr

    if use_vst:
        sample = anscombe_vst(sample) # Only for estimation of clip_val

    clip_val = np.percentile(sample, percentile)
    if not np.isfinite(clip_val) or clip_val <= 0:
        clip_val = float(np.max(sample))
    return float(clip_val)

def preprocess_counts(x, clip_val, use_vst=True, dtype=np.float32):
    """
    Normalization: optional VST -> clip -> /clip -> [0,1]
    """
    x = anscombe_vst(x) if use_vst else x
    x = np.clip(x, 0, clip_val) / clip_val
    return x.astype(dtype)


# %%
def build_sequential_dataset(low_data, high_data, size, group_len, dtype=np.float32):
    """
    Generates training data:
      X: (B, size, H, W) = window of `size` Low-Count images
      Y: (B, H, W)       = Ground truth = middle High-Count image of the window
    """
    assert low_data.shape == high_data.shape, "low/high must have identical shapes"
    N, H, W = low_data.shape

    if size % 2 == 0 or size < 1:
        raise ValueError("`size` must be odd and >= 1 (e.g., 3, 5, 7)")
    if N % group_len != 0:
        raise ValueError(f"N={N} is not a multiple of group_len={group_len}.")

    k = size // 2       # Index for middle of window
    num_groups = N // group_len

    X_list, Y_list = [], []

    for group_index in range(num_groups):
        start = group_index * group_len
        end   = start + group_len
        # slide window inside this block only
        for n in range(start, end - size + 1):
            seq_win    = low_data[n : n + size]   # (size, H, W)
            center_idx = n + k                    # middle frame index
            target     = high_data[center_idx]    # (H, W)
            X_list.append(seq_win)
            Y_list.append(target)

    X = np.stack(X_list, axis=0).astype(dtype)   # (B, size, H, W)
    Y = np.stack(Y_list, axis=0).astype(dtype)   # (B, H, W)
    return X, Y

# ===== Applying on loaded (N,H,W) data =====
size = 5 # Adjust to find best value (e.g. 3,5,7)
group_len = 41

# === Normalisierung ===
USE_VST = True
PCT = 99.9

clip_val = compute_clip_from_high(high_data, percentile=PCT, use_vst=USE_VST)

low_data_n  = preprocess_counts(low_data,  clip_val, use_vst=USE_VST)
high_data_n = preprocess_counts(high_data, clip_val, use_vst=USE_VST)

# Dann wie gewohnt:
X, Y = build_sequential_dataset(low_data_n, high_data_n, size, group_len, dtype=np.float32)

print("X shape (B, size, H, W):", X.shape)
print("Y shape (B, H, W):     ", Y.shape)

# ---- Sanity check ----
b0_start = 0
k = size // 2

seq0 = X[0]                  # (size,H,W) -> should be low_data_n[0:size]
gt0  = Y[0]                  # (H,W)      -> should be high_data_n[k]

# Comparing arrays
print("Seq == low_n[0:5]:",
      np.allclose(seq0, low_data_n[0:size]))
print("GT  == high_n[k]:",
      np.allclose(gt0,  high_data_n[b0_start + k]))


# %%
# ===== Visualization of some samples =====

def show_window_plus_gt(X, Y, sample_idx, size=5, share_scale=False, group_len=41):
    """
    Shows training data samle (low count pictures) + ground truth (high count picture).
    sample_idx: Index in X/Y
    """
    sequence = X[sample_idx]   # (size,H,W)
    gt  = Y[sample_idx]   # (H,W)

    k = size // 2       # Index for middle of window
    group_idx = sample_idx // (group_len - size + 1)
    offset_in_group = sample_idx % (group_len - size + 1)
    global_start = group_idx * group_len + offset_in_group
    frame_indices = list(range(global_start, global_start + size))
    gt_index = global_start + k

    # Plotting low count pictures (window)
    fig, axes = plt.subplots(1, size + 1, figsize=(3*(size+1), 3))
    for j in range(size):
        vmin, vmax = np.percentile( sequence[j].ravel(), (1, 99))
        im = axes[j].imshow( sequence[j], cmap="gray_r", origin="lower", aspect="equal",
                            vmin=vmin, vmax=vmax)
        axes[j].set_title(f"Low (idx={frame_indices[j]})")
        axes[j].axis("off")
        cbar = fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")

    # Plotting high count picture
    vmin, vmax = np.percentile(gt.ravel(), (1, 99))
    im_gt = axes[-1].imshow(gt, cmap="gray_r", origin="lower", aspect="equal",
                            vmin=vmin, vmax=vmax)
    axes[-1].set_title(f"High [GT] (idx={gt_index})")
    axes[-1].axis("off")
    cbar = fig.colorbar(im_gt, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")

    plt.tight_layout()
    plt.show()

# Visualize some samples
for idx in range(0, 4):
    show_window_plus_gt(X, Y, sample_idx=idx, share_scale=False)

