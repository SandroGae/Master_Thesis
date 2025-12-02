#!/usr/bin/env python3
import numpy as np
import h5py
import sys
from tensorflow.keras import models
from pathlib import Path


ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")

# Modelle laden
MODELS = {
    "FILE_25D_3STACK": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras",
    "FILE_25D_5STACK": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras",
    "FILE_3d_middle_improved": "unet_3d_SSIM_middle_improved__seed42__bf64__D3__lossMAE_SSIM__20251112-180318_loss0.0479_val0.0522.keras",
    "FILE_3d_middle_improved_V2": "unet_3d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-144919_loss0.0479_val0.0517.keras",
    "FILE_25d_middle_improved_V2_kernel_3x5": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-103044_loss0.0523_val0.0587.keras",
    "FILE_25d_middle_improved_V2_interpolated": "unet_25d_D5_VarStride1-24__20251201-093031_loss0.0690_val0.0594.keras"
}

CHOSEN_NAME = "FILE_25d_middle_improved_V2_interpolated"
MODEL_FILE = MODELS[CHOSEN_NAME]
MODEL_PATH = ROOT_DIR / "Plots/Unet/Keras" / MODEL_FILE

# Wahl der Serie
SERIES_IDX = 12   
SERIES_IDX_0 = SERIES_IDX - 1
SERIES_LEN = 41

MODEL_PATH   = ROOT_DIR / "Plots/Unet/Keras" / MODEL_FILE
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"
OUT_DIR      = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"

# Depth aus filename extrahieren
if "_D7_" in MODEL_FILE: 
    DEPTH = 7
elif "_D5_" in MODEL_FILE: 
    DEPTH = 5
elif "_D3_" in MODEL_FILE: 
    DEPTH = 3


def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):

    global_start = series_idx_0 * series_len + window_start_idx_0 # Erstes Bild

    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:, :, global_start : global_start + depth]  # (Heigt, Width, Depth)
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]

        low_count_depth_first = np.moveaxis(low_count, -1, 0) # Schiebe Axe (Heigt, Width, Depth) -->  (Depth, Heigt, Width)
        high_count_depth_first = np.moveaxis(high_count, -1, 0)

        low_count_expanded = np.expand_dims(low_count_depth_first, axis=(0, -1)) # Erweitere (Depth, Heigt, Width) --> (Batch, Depth, Heigt, Width, Channel)
        high_count_expanded = np.expand_dims(high_count_depth_first, axis=(0, -1))

    return (low_count_expanded.astype(np.float32), high_count_expanded.astype(np.float32))

def normalize(volume, scale=10000.0):
    volume = np.maximum(volume, 0.0) # Clip [0, infinity]    
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12 # Summiere über Depth, Height, Width
    scaled_volume = (volume / sums) * scale # Skaliere Slicewise mit Faktor 10'000
    return scaled_volume

def main():
    print(f"Modell: {MODEL_FILE}")
    print(f"Tiefe:  {DEPTH}")
    print(f"Erstelle Vorhersagen für GANZE Serie {SERIES_IDX} (Länge {SERIES_LEN})...")
    
    # Modell laden
    model = models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape
    input_dimension = len(input_shape)
    
    # Arrays erstellen für die ganze Serie 
    H, W = 192, 240
    full_stack_lc   = np.zeros((SERIES_LEN, H, W), dtype=np.float32)
    full_stack_pred = np.zeros((SERIES_LEN, H, W), dtype=np.float32)
    full_stack_gt   = np.zeros((SERIES_LEN, H, W), dtype=np.float32)
    

    center_offset = DEPTH // 2
    count_predicted = 0

    for img_idx_0 in range(SERIES_LEN): # 0 bis 40
        window_start = img_idx_0 - center_offset
        if window_start < 0 or (window_start + DEPTH) > SERIES_LEN:
            continue
            
        # Daten laden
        X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, SERIES_IDX_0, window_start, DEPTH)
        X_input = normalize(X_raw)

        # Modell-Typ basierend auf Input-Dimensionen erkennen
        if input_dimension == 4:
            X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) 
        elif input_dimension == 5:
            X_feed = X_input    
        else:
            raise ValueError(f"Fehler: Unbekannte Dimensionen ({input_dimension}). Erwarte 4 oder 5.")

        # Vorhersage
        Y_pred_raw = model.predict(X_feed, verbose=0)
        
        # Mitte extrahieren
        if Y_pred_raw.ndim == 5:
            mid_idx = Y_pred_raw.shape[1] // 2
            img_pred = Y_pred_raw[0, mid_idx, :, :, 0]
        elif Y_pred_raw.ndim == 4:
            img_pred = Y_pred_raw[0, :, :, 0]
        else:
            raise ValueError("Falsche Output Shape vom Modell")
            
        # Referenzbilder holen (Mitte des Input Fensters)
        img_lc = X_input[0, center_offset, :, :, 0]
        img_gt = normalize(Y_raw)[0, center_offset, :, :, 0]
        
        # Ins große Array schreiben
        full_stack_lc[img_idx_0]   = img_lc
        full_stack_pred[img_idx_0] = img_pred
        full_stack_gt[img_idx_0]   = img_gt
        
        count_predicted += 1
        print(f"Bild {img_idx_0 + 1}/{SERIES_LEN} berechnet...", end='\r')

    # Speichern
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / f"Pred_{CHOSEN_NAME}_D{DEPTH}_S{SERIES_IDX}_FullSeries.npz"
    np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
    print(f"Gespeichert: {outfile.name}")

if __name__ == "__main__":
    main()