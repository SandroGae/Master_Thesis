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
    """
    Prüft ob die Datei existiert, lesbar ist UND echte Daten enthält.
    Falls max(pred) == 0 ist, wird die Datei als ungültig markiert.
    """
    if not filepath.exists(): 
        return False
    try:
        with np.load(filepath) as data:
            # Wenn das Modell nur Nullen geliefert hat, ist die Datei wertlos
            if np.max(data['pred']) <= 1e-12:
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
    # 1. Task ID holen
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Modell-Liste sammeln und NUMERISCH sortieren
    # Wichtig: Alphabetische Sortierung würde Point_1, Point_10, Point_11 liefern.
    # Wir sortieren nach der Zahl hinter 'Point_'.
    all_keras_files = list(SCRATCH_DATA.glob("Point_*/*.keras"))
    
    def get_sort_key(path):
        # Extrahiert die Nummer aus 'Point_19...' und den Seed aus dem Filename
        point_num = int(path.parent.name.split('_')[1])
        return (point_num, path.name)

    model_paths = sorted(all_keras_files, key=get_sort_key)
    
    if task_id >= len(model_paths):
        print(f"Task ID {task_id} außerhalb der Range ({len(model_paths)} Modelle gefunden).")
        return

    model_path = model_paths[task_id]
    model_name = model_path.stem # z.B. P19_a0.0833_b0.5000_seed43
    print(f"--- TASK {task_id} ---")
    print(f"Modell: {model_name}")
    print(f"Pfad:   {model_path}")

    # 3. Check welche Serien für dieses Modell noch fehlen oder leer (0.0) sind
    needed_series = [s for s in SERIES_LIST if not is_npz_valid(OUT_DIR / f"Eval_{model_name}_S{s}.npz")]
    
    if not needed_series:
        print(f"Alle Serien für {model_name} bereits valide vorhanden. Überspringe.")
        return

    # 4. Modell laden
    print(f"Lade Modell...")
    model = models.load_model(model_path, compile=False)
    input_shape = model.input_shape
    # Bestimme Tiefe (depth) für 4D oder 5D Inputs
    depth = input_shape[-1] if len(input_shape) == 4 else input_shape[1]
    center_offset = depth // 2

    # 5. Inference Loop
    for series_idx in needed_series:
        outfile = OUT_DIR / f"Eval_{model_name}_S{series_idx}.npz"
        print(f"  -> Berechne Serie S{series_idx}...")
        
        series_idx_0 = series_idx - 1
        full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

        for img_idx_0 in range(SERIES_LEN):
            window_start = img_idx_0 - center_offset
            
            # Padding-Ersatz: Nur berechnen wenn Fenster im Volume liegt
            if window_start < 0 or (window_start + depth) > SERIES_LEN: 
                continue
            
            # Daten laden & normalisieren
            X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, depth, SERIES_LEN)
            X_input = normalize(X_raw)
            
            # Channel-Order Fix für 2D-Unet (falls nötig)
            X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) if len(input_shape) == 4 else X_input
            
            # Vorhersage
            Y_pred_raw = model.predict(X_feed, verbose=0)
            
            # Mittleren Slice extrahieren
            if Y_pred_raw.ndim == 4: # [1, H, W, 1]
                img_pred = Y_pred_raw[0, :, :, 0]
            else: # [1, D, H, W, 1]
                img_pred = Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
            
            # Speichern
            full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
            full_stack_pred[img_idx_0] = img_pred
            full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

        # Speichern der Ergebnisse
        np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
        print(f"     OK: {outfile.name} gespeichert.")

    # Cleanup um GPU-Speicher für den nächsten Task im Array freizugeben (falls zutreffend)
    del model
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    main()