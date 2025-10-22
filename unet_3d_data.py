# unet_3d_data.py
# ALT-PIPELINE: Keine Anscombe/VST. Globales 99.9-Perzentil der High-Counts, dann Clip & Scale auf [0,1].
# Baut 5-zu-5 (size-zu-size) 3D-Samples fuer Training/Val/Test.

import gc
import h5py
import numpy as np
from pathlib import Path

__all__ = [
    "prepare_in_memory",
    "build_sequential_dataset",
    "compute_clip_from_high",
    "preprocess_counts",
]

def compute_clip_from_high(high_data, percentile=99.9, max_samples=5_000_000, rng=None):
    """
    Bestimmt einen globalen Clip-Wert aus den High-Count-Daten (kein VST).
    """
    rng = np.random.default_rng() if rng is None else rng
    arr = high_data.ravel()
    sample = arr if arr.size <= max_samples else arr[rng.choice(arr.size, size=max_samples, replace=False)]
    clip_val = np.percentile(sample, percentile)
    if not np.isfinite(clip_val) or clip_val <= 0:
        clip_val = float(np.max(sample))
    return float(clip_val)

def preprocess_counts(x, clip_val, dtype=np.float32):
    """
    ALT: Clip im Originalraum und Skalieren auf [0,1].
    """
    x = np.clip(x, 0, clip_val) / clip_val
    return x.astype(dtype)

def build_sequential_dataset(low_data, high_data, size, group_len, dtype=np.float32):
    """
    Erzeugt 3D-Sequenzen:
      X: (B, size, H, W, 1)
      Y: (B, size, H, W, 1)
    """
    assert low_data.shape == high_data.shape, "low/high must have identical shapes"
    N, H, W = low_data.shape
    if size % 2 == 0 or size < 1:
        raise ValueError("`size` must be odd and >= 1 (e.g., 3, 5, 7)")
    if N % group_len != 0:
        raise ValueError(f"N={N} is not a multiple of group_len={group_len}.")

    X_list, Y_list = [], []
    num_groups = N // group_len
    for gidx in range(num_groups):
        start = gidx * group_len
        end   = start + group_len
        for n in range(start, end - size + 1):
            X_list.append(low_data[n:n+size])   # (size,H,W)
            Y_list.append(high_data[n:n+size])  # (size,H,W)

    X = np.stack(X_list, axis=0).astype(dtype)
    Y = np.stack(Y_list, axis=0).astype(dtype)
    X = X[..., None]  # Kanal-Dimension
    Y = Y[..., None]
    return X, Y

def _load_pair(fp):
    with h5py.File(fp, "r") as f:
        high = f["/high_count/data"][:].transpose(2,0,1)  # (N,H,W)
        low  = f["/low_count/data"][:].transpose(2,0,1)   # (N,H,W)
    return high, low

def prepare_in_memory(
    data_dir=Path.home() / "data" / "original_data",
    size=5,
    group_len=41,
    percentile=99.9,
    dtype=np.float32,
):
    """
    ALT: Keine VST. Clip-Wert aus TRAIN/High berechnen, alle Splits (low/high) damit clippen & skalieren.
    Gibt (results, meta) zurueck.
    """
    data = {
        "train": _load_pair(data_dir / "training_data.hdf5"),
        "test":  _load_pair(data_dir / "test_data.hdf5"),
        "val":   _load_pair(data_dir / "validation_data.hdf5"),
    }

    high_train, _ = data["train"]
    clip_val_train = compute_clip_from_high(high_train, percentile=percentile)

    results = {}
    for split in ["train", "test", "val"]:
        high_split, low_split = data[split]
        low_n  = preprocess_counts(low_split,  clip_val_train, dtype=dtype)
        high_n = preprocess_counts(high_split, clip_val_train, dtype=dtype)
        X, Y = build_sequential_dataset(low_n, high_n, size=size, group_len=group_len, dtype=dtype)
        results[split] = (X, Y)
        del low_n, high_n
        gc.collect()

    meta = {
        "pipeline": "ALT_no_VST",
        "size": size,
        "group_len": group_len,
        "percentile": percentile,
        "clip_val": float(clip_val_train),
        "dtype": str(dtype),
    }
    return results, meta

