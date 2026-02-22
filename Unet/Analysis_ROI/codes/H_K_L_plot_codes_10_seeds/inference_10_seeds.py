#!/usr/bin/env python3
import os
import gc
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
from tqdm import tqdm

# =====================================================
# 1. KONFIGURATION
# =====================================================
SCRATCH_DATA  = Path.home() / "scratch" / "43_Models_10_Seeds"
H5_TEST_PATH  = Path.home() / "data" / "original_data" / "test_data.hdf5"
OUT_DIR       = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"

# Die 10 Serien für den Test-Datensatz
SERIES_LIST   = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] 
SERIES_LEN    = 41 # Jede Serie hat 41 Bilder

def is_npz_valid(filepath):
    if not filepath.exists(): 
        return False
    try:
        with np.load(filepath) as data:
            if np.max(data['pred']) <= 1e-12:
                return False
        return True
    except:
        return False

def normalize(volume, scale=10000.0):
    """ Normalisiert ein Volumen basierend auf der Summe seiner Pixel """
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume) + 1e-12
    return (volume / sums) * scale

def main():
    # 1. Task ID von Slurm holen (0-429)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Alle 430 Modelle sammeln und numerisch sortieren
    all_keras_files = list(SCRATCH_DATA.glob("Point_*/*_01_model.keras"))
    
    def get_sort_key(path):
        # Sortiert erst nach Point-Nummer, dann nach Seed
        point_num = int(path.parent.name.split('_')[1])
        seed_num = int(path.name.split('seed')[1][:2])
        return (point_num, seed_num)

    model_paths = sorted(all_keras_files, key=get_sort_key)
    
    if task_id >= len(model_paths):
        print(f"Task ID {task_id} außerhalb der Range ({len(model_paths)} Modelle gefunden).")
        return

    model_path = model_paths[task_id]
    # Extrahiere Namen für die Datei (Pxx_axxxx_bxxxx_seedxx)
    model_name = model_path.name.replace("_01_model.keras", "")
    print(f"\n{'='*60}\nTASK {task_id}: {model_name}\n{'='*60}")

    # 3. Check welche Serien für dieses Modell fehlen
    needed_series = [s for s in SERIES_LIST if not is_npz_valid(OUT_DIR / f"Eval_{model_name}_S{s}.npz")]
    
    if not needed_series:
        print(f"✅ Alle 10 Serien bereits vorhanden. Fertig.")
        return

    # 4. Modell laden
    print(f"Lade Modell von: {model_path.parent.name}")
    model = models.load_model(model_path, compile=False)
    input_shape = model.input_shape # z.B. (None, 192, 240, 5)
    depth = input_shape[-1] if len(input_shape) == 4 else input_shape[1]
    center_offset = depth // 2

    # 5. Inferenz über die Serien
    for series_idx in tqdm(needed_series, desc="Serien-Fortschritt", unit="Serie"):
        outfile = OUT_DIR / f"Eval_{model_name}_S{series_idx}.npz"
        
        series_idx_0 = series_idx - 1
        full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
        full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

        # SPEED-FIX: Lade die komplette Serie (41 Bilder) einmalig in den RAM
        with h5py.File(H5_TEST_PATH, "r") as f:
            start_global = series_idx_0 * SERIES_LEN
            end_global   = (series_idx_0 + 1) * SERIES_LEN
            raw_series_lc = np.moveaxis(f["low_count/data"][:, :, start_global:end_global], -1, 0)
            raw_series_hc = np.moveaxis(f["high_count/data"][:, :, start_global:end_global], -1, 0)

        # Innerer Loop für die 41 Slices (Sliding Window)
        for img_idx_0 in range(SERIES_LEN):
            window_start = img_idx_0 - center_offset
            
            # Nur berechnen, wenn das Fenster (z.B. 5 Slices) komplett in die 41 Bilder passt
            if window_start >= 0 and (window_start + depth) <= SERIES_LEN:
                # Fenster ausschneiden [depth, 192, 240]
                vol_raw = raw_series_lc[window_start : window_start + depth]
                
                # Normalisierung & Vorbereitung für das Modell
                # Modell erwartet: [1, 192, 240, 5]
                vol_norm = normalize(vol_raw.astype(np.float32))
                vol_input = np.transpose(vol_norm, (1, 2, 0))[np.newaxis, ...]
                
                # Vorhersage
                prediction = model.predict(vol_input, verbose=0)
                
                # Falls das Modell mu und sigma liefert (2 Kanäle), nimm mu (0)
                if prediction.shape[-1] > 1:
                    img_pred = prediction[0, ..., 0]
                else:
                    img_pred = np.squeeze(prediction)

                full_stack_pred[img_idx_0] = img_pred

            # Ground Truth und Input werden zur Kontrolle immer gespeichert (normalisiert)
            full_stack_lc[img_idx_0] = raw_series_lc[img_idx_0] / (np.sum(raw_series_lc[img_idx_0]) + 1e-12) * 10000.0
            full_stack_gt[img_idx_0] = raw_series_hc[img_idx_0] / (np.sum(raw_series_hc[img_idx_0]) + 1e-12) * 10000.0

        # Speichern als komprimierte NPZ
        np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)

    # Speicher aufräumen
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"✅ Inferenz für {model_name} abgeschlossen.")

if __name__ == "__main__":
    main()