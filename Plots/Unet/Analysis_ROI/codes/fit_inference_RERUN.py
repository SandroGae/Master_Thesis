#!/usr/bin/env python3
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
from tqdm import tqdm

# GPU Optimierung
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# =====================================================
# 1. KONFIGURATION
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
# Hier liegen deine umbenannten Rank_XX Modelle
MODEL_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "keras_models_RERUN"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
# Neuer Output-Ordner
OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw_RERUN"

# Wir laden alle Serien, die in deinem Plotting-Skript definiert sind
SERIES_LIST = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] 
SERIES_LEN = 41

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):
    global_start = series_idx_0 * series_len + window_start_idx_0 
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:, :, global_start : global_start + depth]
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]
        
        low_count_depth_first = np.moveaxis(low_count, -1, 0)
        high_count_depth_first = np.moveaxis(high_count, -1, 0)
        
        low_count_expanded = np.expand_dims(low_count_depth_first, axis=(0, -1))
        high_count_expanded = np.expand_dims(high_count_depth_first, axis=(0, -1))
    return (low_count_expanded.astype(np.float32), high_count_expanded.astype(np.float32))

def normalize(volume, scale=10000.0):
    volume = np.maximum(volume, 0.0)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    return (volume / sums) * scale

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Alle Modelle finden, die mit "Rank_" beginnen
    model_files = sorted(list(MODEL_DIR.rglob("Rank_*.keras")))
    
    if not model_files:
        print(f"Fehler: Keine Modelle in {MODEL_DIR} gefunden!")
        return

    print(f"Starte Inference für {len(model_files)} Modelle...")

    for model_path in model_files:
        # Extrahiere "Rank_XX" aus dem Dateinamen (z.B. Rank_01)
        rank_name = model_path.name.split("__")[0] 
        
        print(f"\n[MODEL] Verarbeite {rank_name}...")
        
        try:
            model = models.load_model(model_path, compile=False)
            input_dimension = len(model.input_shape)
            # Dynamische Tiefenerkennung aus dem Modell-Input
            depth = model.input_shape[3] if input_dimension == 4 else model.input_shape[1]
        except Exception as e:
            print(f"Fehler beim Laden von {model_path.name}: {e}")
            continue

        for series_idx in SERIES_LIST:
            series_idx_0 = series_idx - 1
            
            full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

            center_offset = depth // 2

            for img_idx_0 in tqdm(range(SERIES_LEN), desc=f"  Serie {series_idx}", leave=False):
                window_start = img_idx_0 - center_offset
                
                # Padding-Logik: Wir überspringen Ränder, wo kein volles Fenster existiert
                if window_start < 0 or (window_start + depth) > SERIES_LEN:
                    continue
                
                X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, depth)
                X_input = normalize(X_raw)

                # Format-Check für 2.5D vs 3D
                if input_dimension == 4:
                    X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) 
                else:
                    X_feed = X_input

                Y_pred_raw = model.predict(X_feed, verbose=0)
                
                # Clipping auf [0, 1] für Konsistenz
                Y_pred_raw = np.clip(Y_pred_raw, 0.0, 1.0)
                Y_gt_norm = np.clip(normalize(Y_raw), 0.0, 1.0)

                # Zentrales Bild extrahieren
                if Y_pred_raw.ndim == 5:
                    img_pred = Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
                else:
                    img_pred = Y_pred_raw[0, :, :, 0]
                
                full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
                full_stack_pred[img_idx_0] = img_pred
                full_stack_gt[img_idx_0]   = Y_gt_norm[0, center_offset, :, :, 0]

            # Speichern im neuen Format
            outfile = OUT_DIR / f"Pred_{rank_name}_D{depth}_S{series_idx}_FullSeries.npz"
            np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
            
        # Speicher freigeben für das nächste Modell
        tf.keras.backend.clear_session()

    print(f"\nInference beendet. Dateien liegen in: {OUT_DIR}")

if __name__ == "__main__":
    main()