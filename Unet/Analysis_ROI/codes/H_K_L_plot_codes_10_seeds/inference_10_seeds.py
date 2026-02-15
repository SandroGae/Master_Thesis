#!/usr/bin/env python3
import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path

# =====================================================
# 1. KONFIGURATION (Dein spezifischer Zielordner)
# =====================================================
SCRATCH_DATA = Path.home() / "scratch" / "43_Models_10_Seeds"
H5_TEST_PATH = Path.home() / "data" / "original_data" / "test_data.hdf5"
OUT_DIR = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"

# Welche Serien sollen evaluiert werden?
SERIES_LIST = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] 
SERIES_LEN  = 41

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):
    global_start = series_idx_0 * series_len + window_start_idx_0 
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:, :, global_start : global_start + depth]
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]
        
    low_count_depth_first  = np.moveaxis(low_count, -1, 0)
    high_count_depth_first = np.moveaxis(high_count, -1, 0)
    
    # Batch & Channel Dimension hinzufügen
    lc_exp = np.expand_dims(low_count_depth_first, axis=(0, -1))
    hc_exp = np.expand_dims(high_count_depth_first, axis=(0, -1))
    return (lc_exp.astype(np.float32), hc_exp.astype(np.float32))

def normalize(volume, scale=10000.0):
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    return (volume / sums) * scale

# =====================================================
# 3. MAIN PIPELINE
# =====================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Automatische Suche nach allen .h5 Dateien in den Point-Ordnern
    # Wir suchen nach *.h5, da diese in deinem Beast-Code die finalen Modelle sind
    model_paths = sorted(list(SCRATCH_DATA.glob("Point_*/*.h5")))
    
    print(f"Gefundene Modelle: {len(model_paths)}")
    
    for model_path in model_paths:
        model_name = model_path.stem # z.B. P11_a0.0000_b0.9167_seed43
        print(f"\n>>> Verarbeite Modell: {model_name}")
        
        # Modell laden (compile=False spart Zeit und verhindert Fehler bei Custom Losses)
        try:
            model = models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"Fehler beim Laden von {model_name}: {e}")
            continue

        input_dimension = len(model.input_shape)
        # Tiefe erkennen (Standard 5, falls nicht anders im Namen)
        depth = 5
        if "_D7_" in model_name: depth = 7
        elif "_D3_" in model_name: depth = 3

        # Innere Schleife: Serien evaluieren
        for series_idx in SERIES_LIST:
            series_idx_0 = series_idx - 1
            
            full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

            center_offset = depth // 2

            for img_idx_0 in range(SERIES_LEN):
                window_start = img_idx_0 - center_offset
                if window_start < 0 or (window_start + depth) > SERIES_LEN:
                    continue
                
                # Daten laden & normalisieren
                X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, depth)
                X_input = normalize(X_raw)

                # Format anpassen (2.5D vs 3D)
                if input_dimension == 4: # 2.5D: (Batch, H, W, Depth)
                    X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) 
                else: # 3D: (Batch, Depth, H, W, 1)
                    X_feed = X_input

                # Prediction
                Y_pred_raw = model.predict(X_feed, verbose=0)
                
                # Mittleres Slice extrahieren
                if Y_pred_raw.ndim == 5:
                    img_pred = Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
                else:
                    img_pred = Y_pred_raw[0, :, :, 0]
                
                full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
                full_stack_pred[img_idx_0] = img_pred
                full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

            # Speichern als .npz
            outfile = OUT_DIR / f"Eval_{model_name}_S{series_idx}.npz"
            np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
            
        # Session aufräumen um Speicherlecks zu verhindern
        tf.keras.backend.clear_session()

    print("\nFertig! Alle Evaluationen abgeschlossen.")

if __name__ == "__main__":
    main()