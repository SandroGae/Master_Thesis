#!/usr/bin/env python3
import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path

# =====================================================
# 1. KONFIGURATION
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
# Dein neuer Test-Ordner für das heruntergeladene Modell
MODEL_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "KERAS_MODEL" / "keras_TEST"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "Predictions_Raw"

# Das Modell-Dictionary mit allen drei heute trainierten Varianten
MODELS = {
    "Beast_P00_S42": "P00_a0.0000_b0.0000_seed42_best_model.keras",
    "Beast_P00_S43": "P00_a0.0000_b0.0000_seed43_best_model.keras",
    "Beast_P00_S44": "P00_a0.0000_b0.0000_seed44_best_model.keras",
}

SERIES_LIST = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] 
SERIES_LEN = 41
DEPTH = 5 # Festgelegt für dieses Modell

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):
    global_start = series_idx_0 * series_len + window_start_idx_0 
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:, :, global_start : global_start + depth]
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]
        
        # Umwandeln in (Depth, H, W)
        low_count_depth_first = np.moveaxis(low_count, -1, 0)
        high_count_depth_first = np.moveaxis(high_count, -1, 0)
        
        # Batch- und Channel-Dimension hinzufügen: (1, Depth, H, W, 1)
        low_count_expanded = np.expand_dims(low_count_depth_first, axis=(0, -1))
        high_count_expanded = np.expand_dims(high_count_depth_first, axis=(0, -1))
    return (low_count_expanded.astype(np.float32), high_count_expanded.astype(np.float32))

def normalize(volume, scale=10000.0):
    # ReLU-Äquivalent: Negative Werte kappen
    volume = np.maximum(volume, 0.0)
    # Summennormierung pro Slice (analog zum Training)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    return (volume / sums) * scale

# =====================================================
# 3. MAIN INFERENCE LOOP
# =====================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for chosen_name, model_file in MODELS.items():
        model_path = MODEL_DIR / model_file
        
        if not model_path.exists():
            print(f"Fehler: {model_file} nicht gefunden in {MODEL_DIR}")
            continue

        print(f"\n--- Lade Modell: {chosen_name} ---")
        # compile=False ist wichtig, da wir nur Inferenz machen und keine Custom-Losses brauchen
        model = models.load_model(model_path, compile=False)
        
        # Bestimmen, ob das Modell 4D (H,W,C) oder 5D (B,D,H,W,C) Input erwartet
        # Dein 2.5D Modell erwartet (H, W, Depth) als Input-Format
        is_25d_format = len(model.input_shape) == 4 

        for series_idx in SERIES_LIST:
            series_idx_0 = series_idx - 1
            print(f"Verarbeite Serie {series_idx}...")
            
            full_stack_lc   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_pred = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)
            full_stack_gt   = np.zeros((SERIES_LEN, 192, 240), dtype=np.float32)

            center_offset = DEPTH // 2

            for img_idx_0 in range(SERIES_LEN):
                window_start = img_idx_0 - center_offset
                
                # Nur Slices berechnen, für die ein volles Fenster existiert
                if window_start < 0 or (window_start + DEPTH) > SERIES_LEN:
                    continue
                
                # Daten laden & normalisieren
                X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, series_idx_0, window_start, DEPTH)
                X_norm = normalize(X_raw)

                # Format-Konvertierung für 2.5D (H, W, Depth als Channels)
                if is_25d_format:
                    # (1, D, H, W, 1) -> (1, H, W, D)
                    X_feed = np.transpose(np.squeeze(X_norm, axis=-1), (0, 2, 3, 1))
                else:
                    X_feed = X_norm

                # Inferenz
                prediction = model.predict(X_feed, verbose=0)
                
                # Ergebnis extrahieren (Center Slice)
                if prediction.ndim == 5: # Falls 3D Modell
                    img_pred = prediction[0, prediction.shape[1]//2, :, :, 0]
                else: # 2.5D Modell (liefert direkt 2D Resultat)
                    img_pred = prediction[0, :, :, 0]
                
                # Stacks füllen
                full_stack_lc[img_idx_0]   = X_norm[0, center_offset, :, :, 0]
                full_stack_pred[img_idx_0] = img_pred
                full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

            # Speichern als NPZ
            outfile = OUT_DIR / f"Pred_{chosen_name}_S{series_idx}.npz"
            np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
            print(f"Datei erstellt: {outfile.name}")

    print("\nInferenz abgeschlossen.")

if __name__ == "__main__":
    main()