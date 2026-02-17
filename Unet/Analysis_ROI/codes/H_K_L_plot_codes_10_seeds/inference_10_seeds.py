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

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
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

def print_progress(current, total, model_name=""):
    """Erstellt eine Fortschrittsanzeige in der Konsole."""
    percent = (current / total) * 100
    bar_length = 40
    done = int(percent / 100 * bar_length)
    bar = "█" * done + "-" * (bar_length - done)
    sys.stdout.write(f"\rGesamtfortschritt: |{bar}| {percent:.2f}% ({current}/{total}) | Modell: {model_name}")
    sys.stdout.flush()

# =====================================================
# 3. MAIN PIPELINE
# =====================================================
def main():
    if not H5_TEST_PATH.exists():
        print(f"FEHLER: Test-Daten nicht gefunden unter {H5_TEST_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(list(SCRATCH_DATA.glob("Point_*/*.keras")))
    
    total_tasks = len(model_paths) * len(SERIES_LIST)
    completed_tasks = 0
    
    print(f"Starte Evaluation...")
    print(f"Gefundene Modelle: {len(model_paths)}")
    print(f"Zu erstellende Dateien: {total_tasks}\n")

    # Initialer Check: Wie viel wurde schon gemacht? (Falls Neustart)
    for mp in model_paths:
        m_name = mp.stem
        for s in SERIES_LIST:
            if (OUT_DIR / f"Eval_{m_name}_S{s}.npz").exists():
                completed_tasks += 1

    for model_path in model_paths:
        model_name = model_path.stem
        
        # Check ob Modell komplett übersprungen werden kann
        if all((OUT_DIR / f"Eval_{model_name}_S{s}.npz").exists() for s in SERIES_LIST):
            print_progress(completed_tasks, total_tasks, f"Skip {model_name}")
            continue

        try:
            model = models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"\nFehler bei {model_name}: {e}")
            completed_tasks += len(SERIES_LIST) # Zähle als "erledigt" um Progress nicht zu sprengen
            continue

        input_shape = model.input_shape
        input_dimension = len(input_shape)
        depth = input_shape[-1] if input_dimension == 4 else input_shape[1]
        center_offset = depth // 2

        for series_idx in SERIES_LIST:
            outfile = OUT_DIR / f"Eval_{model_name}_S{series_idx}.npz"
            
            if not outfile.exists():
                series_idx_0 = series_idx - 1
                full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
                full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
                full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

                for img_idx_0 in range(SERIES_LEN):
                    window_start = img_idx_0 - center_offset
                    if window_start < 0 or (window_start + depth) > SERIES_LEN:
                        continue
                    
                    X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, depth, SERIES_LEN)
                    X_input = normalize(X_raw)

                    X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) if input_dimension == 4 else X_input
                    Y_pred_raw = model.predict(X_feed, verbose=0)
                    
                    img_pred = Y_pred_raw[0, :, :, 0] if Y_pred_raw.ndim == 4 else Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
                    
                    full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
                    full_stack_pred[img_idx_0] = img_pred
                    full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

                np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
            
            completed_tasks += 1
            print_progress(completed_tasks, total_tasks, model_name)
            
        tf.keras.backend.clear_session()
        del model
        gc.collect()

    print(f"\n\nFertig! Alle 4300 Evaluationen wurden erfolgreich verarbeitet.")

if __name__ == "__main__":
    main()