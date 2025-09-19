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
# ====== Imports ===
import h5py
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt
import gc
import os


# %%
# ================= Functions for Normalizing Data =================
def anscombe_vst(x):
    """
    Negative values get yeeted to zero (counts should not be negative)
    """
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
    sample = arr if arr.size <= max_samples else arr[rng.choice(arr.size, size=max_samples, replace=False)]
    if use_vst:
        sample = anscombe_vst(sample)
    clip_val = np.percentile(sample, percentile)
    if not np.isfinite(clip_val) or clip_val <= 0:
        clip_val = float(np.max(sample))
    return float(clip_val)

def preprocess_counts(x, clip_val, use_vst=True, dtype=np.float32):
    """
    Normalization: optional VST -> clip -> /clip -> [0,1]
    """
    if use_vst:
        x = anscombe_vst(x)
    x = np.clip(x, 0, clip_val) / clip_val
    return x.astype(dtype)


# %%

# ================= Function for building 3D Datasets =================

def build_sequential_dataset(low_data, high_data, size, group_len, dtype=np.float32):
    """
    Generates training data:
      X: (B, size, H, W) = window of `size` Low-Count images
      Y: (B, size, H, W) = Ground truth = window of `size` High-Count images (3D output)
      N: Number of Pictrues in total
      H: Height of each image
      W: Width of each image
      size: Size of sliding window (must be odd)
      group_len: Length of each block (41 for training/test/val)
    """
    assert low_data.shape == high_data.shape, "low/high must have identical shapes"
    N, H, W = low_data.shape
    if size % 2 == 0 or size < 1:
        raise ValueError("`size` must be odd and >= 1 (e.g., 3, 5, 7)")
    if N % group_len != 0:
        raise ValueError(f"N={N} is not a multiple of group_len={group_len}.")

    num_groups = N // group_len
    X_list, Y_list = [], []

    for group_index in range(num_groups):
        start = group_index * group_len
        end   = start + group_len
        # slide window inside this block only
        for n in range(start, end - size + 1):         # stride = 1
            X_list.append(low_data[n: n + size])       # (size,H,W)
            Y_list.append(high_data[n: n + size])      # (size,H,W)

    X = np.stack(X_list, axis=0).astype(dtype)   # (B, size, H, W)
    Y = np.stack(Y_list, axis=0).astype(dtype)   # (B, size, H, W)
    # Adding Channel-Dimension since PyTorch expects (B,C,D,H,W) with C=1
    X = X[..., None]  # (B, size, H, W, 1)
    Y = Y[..., None]  # (B, size, H, W, 1)
    return X, Y


# %%
def prepare_in_memory_5to5(
    data_dir=Path("data") / "original_data",
    size=5,
    group_len=41,
    use_vst=False, # Activates / Deactivates Anscombe transform
    percentile=99.9,
    dtype=np.float32,
):
    def _load(fp):
        with h5py.File(fp, "r") as f:
            high = f["/high_count/data"][:].transpose(2,0,1)
            low  = f["/low_count/data"][:].transpose(2,0,1)
        return high, low

    data = {
        "train": _load(data_dir / "training_data.hdf5"),
        "test":  _load(data_dir / "test_data.hdf5"),
        "val":   _load(data_dir / "validation_data.hdf5"),
    }

    high_train, _ = data["train"]
    clip_val_train = compute_clip_from_high(high_train, percentile=percentile, use_vst=use_vst)

    results = {}
    for split in ["train", "test", "val"]:
        high_split, low_split = data[split]
        low_n  = preprocess_counts(low_split,  clip_val_train, use_vst=use_vst, dtype=dtype)
        high_n = preprocess_counts(high_split, clip_val_train, use_vst=use_vst, dtype=dtype)
        X, Y = build_sequential_dataset(low_n, high_n, size=size, group_len=group_len, dtype=dtype)
        results[split] = (X, Y)
        del low_n, high_n
        gc.collect()

    return results, size


# %%
"""
# ===== Visualization of some samples =====

def show_window_pair_3d(X, Y, sample_idx, size=5, group_len=41):

    seq_low  = X[sample_idx, ..., 0]   # (size,H,W)
    seq_high = Y[sample_idx, ..., 0]   # (size,H,W)

    k = size // 2
    group_idx = sample_idx // (group_len - size + 1)
    offset_in_group = sample_idx % (group_len - size + 1)
    global_start = group_idx * group_len + offset_in_group
    frame_indices = list(range(global_start, global_start + size))

    fig, axes = plt.subplots(2, size, figsize=(3 * size, 6))
    for j in range(size):
        v1, v2 = np.percentile(seq_low[j].ravel(), (1,99))
        im_low = axes[0, j].imshow(seq_low[j], cmap="gray_r", origin="lower",
                                   aspect="equal", vmin=v1, vmax=v2)
        axes[0, j].set_title(f"Low idx={frame_indices[j]}"); axes[0, j].axis("off")
        fig.colorbar(im_low, ax=axes[0, j], fraction=0.046, pad=0.04)

        v1, v2 = np.percentile(seq_high[j].ravel(), (1,99))
        im_high = axes[1, j].imshow(seq_high[j], cmap="gray_r", origin="lower",
                                    aspect="equal", vmin=v1, vmax=v2)
        axes[1, j].set_title(f"High idx={frame_indices[j]}"); axes[1, j].axis("off")
        fig.colorbar(im_high, ax=axes[1, j], fraction=0.046, pad=0.04)

    axes[0, k].set_title(f"Low idx={frame_indices[k]} (center)")
    axes[1, k].set_title(f"High idx={frame_indices[k]} (center)")
    plt.tight_layout(); plt.show()

# Visualizing examples:
X_vis, Y_vis = results["train"]
for idx in range(3):
    show_window_pair_3d(X_vis, Y_vis, sample_idx=idx, size=size, group_len=group_len)
"""
