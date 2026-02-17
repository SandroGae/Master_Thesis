#!/usr/bin/env python3
import os
import gc
import sys
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path

# =====================================================
# 1. KONFIGURATION
# =====================================================
SCRATCH_DATA  = Path.home() / "scratch" / "43_Models_10_Seeds"
H5_TEST_PATH  = Path.home() / "data" / "original_data" / "test_data.hdf5"
OUT_DIR       = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"
SERIES_LIST   = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] 
SERIES_LEN    = 41

def is_npz_valid(filepath):
    if not filepath.exists(): return False
    try:
        with np.load(filepath) as data:
            # NEU: Prüfe, ob die Prediction wirklich Daten enthält (nicht nur Nullen)
            if np.max(data['pred']) == 0:
                return False 
        return True
    except:
        return False

def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):
    global_start = series_idx_0 * series_len + window_start_idx_0 
    with h5py.File(h5_path, "r") as f:
        low_count  = f["low_count/data"][:, :, global_start : global_start + depth]
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]
    lc_exp = np.expand_dims(np.moveaxis(low_count, -1, 0), axis=(0, -1))
    hc_exp = np.expand_dims(np.moveaxis(high_count, -1, 0), axis=(0, -1))
    return (lc_exp.astype(np.float32), hc_exp.astype(np.float32))

def normalize(volume, scale=10000.0):
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    return (volume / sums) * scale

def main():
    # Hole die Modell-Nummer vom Slurm Array Index
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(list(SCRATCH_DATA.glob("Point_*/*.keras")))
    
    if task_id >= len(model_paths):
        print(f"Task ID {task_id} außerhalb der Range.")
        return

    model_path = model_paths[task_id]
    model_name = model_path.stem
    print(f"TASK {task_id}: Starte Modell {model_name}")

    # Check ob das Modell überhaupt bearbeitet werden muss
    needed_series = [s for s in SERIES_LIST if not is_npz_valid(OUT_DIR / f"Eval_{model_name}_S{s}.npz")]
    
    if not needed_series:
        print(f"Modell {model_name} bereits vollständig und valide. Überspringe.")
        return

    # Modell laden
    model = models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    depth = input_shape[-1] if len(input_shape) == 4 else input_shape[1]
    center_offset = depth // 2

    for series_idx in needed_series:
        outfile = OUT_DIR / f"Eval_{model_name}_S{series_idx}.npz"
        print(f"  -> Verarbeite Serie {series_idx}")
        
        series_idx_0 = series_idx - 1
        full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

        for img_idx_0 in range(SERIES_LEN):
            window_start = img_idx_0 - center_offset
            if window_start < 0 or (window_start + depth) > SERIES_LEN: continue
            
            X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, depth, SERIES_LEN)
            X_input = normalize(X_raw)
            X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) if len(input_shape) == 4 else X_input
            
            Y_pred_raw = model.predict(X_feed, verbose=0)
            img_pred = Y_pred_raw[0, :, :, 0] if Y_pred_raw.ndim == 4 else Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
            
            full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
            full_stack_pred[img_idx_0] = img_pred
            full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

        np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)

if __name__ == "__main__":
    main()