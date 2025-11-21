#!/usr/bin/env python3
import numpy as np
import h5py
import sys
from tensorflow.keras import models
from pathlib import Path


ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")

# Modelle laden
FILE_25D_3STACK = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras"
FILE_25D_5STACK = "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras"
FILE_3d_middle_improved = "unet_3d_SSIM_middle_improved__seed42__bf64__D3__lossMAE_SSIM__20251112-180318_loss0.0479_val0.0522.keras"
FILE_3d_middle_improved_V2 = "unet_3d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-144919_loss0.0479_val0.0517.keras"

MODEL_FILE = FILE_3d_middle_improved_V2

SERIES_IDX = 12  # Serie wählen
IMAGE_IDX = 19  # Bildwahl 1 basiert
SERIES_IDX_0 = SERIES_IDX -1
IMAGE_IDX_0 = IMAGE_IDX - 1

MODEL_PATH   = ROOT_DIR / "Plots/Unet/Keras" / MODEL_FILE
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"
OUT_DIR      = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
SERIES_LEN   = 41

# Depth aus filename extrahieren
if "_D7_" in MODEL_FILE: DEPTH = 7
elif "_D5_" in MODEL_FILE: DEPTH = 5
elif "_D3_" in MODEL_FILE: DEPTH = 3
else: print("Depth nicht gefunden!")


# Funktionen
def load_volume_by_start_index(h5_path, series_idx_0, window_start_idx_0, depth, series_len=41):
    """
    Input: low count and high count (Height, Width, Depth)
    Reads data from hdf5 files and reshapes them to (Batch = 1, Depth, Height, Width, Channel = 1)
    """
    global_start = series_idx_0 * series_len + window_start_idx_0
    
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:, :, global_start : global_start + depth] # (Height, Width, Depth)
        high_count = f["high_count/data"][:, :, global_start : global_start + depth]

        low_count_depth_first = np.moveaxis(low_count, -1, 0)   # (Height, Width, Depth) --> (Depth, Height, Width)
        high_count_depth_first = np.moveaxis(high_count, -1, 0)

        low_count_expanded = np.expand_dims(low_count_depth_first, axis=(0, -1))    # (Depth, Height, Width) --> (Batch = 1, Depth, Height, Width, Channel = 1)
        high_count_expanded = np.expand_dims(high_count_depth_first, axis=(0, -1))

    return (low_count_expanded.astype(np.float32), high_count_expanded.astype(np.float32))

def normalize(volume, scale=10000.0):
    """
    Input: Volumes of size (Batch = 1, Depth, Height, Width, Channel = 1)
    Clips negative values and normalizes volumes slice wise
    """
    volume = np.maximum(volume, 0.0)    # Clip (0,infinity)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12 # Summe über Height, Width, Depth
    scaled_volume = (volume / sums) * scale
    return scaled_volume

def main():
    print(f"Modell: {MODEL_FILE}")
    print(f"Tiefe:  {DEPTH}")
    print(f"Ziel:   Serie {SERIES_IDX}, Bild {IMAGE_IDX}")

    center_offset = DEPTH // 2
    start_idx_0 = IMAGE_IDX_0 - center_offset # Start index des Fensters

    if start_idx_0 < 0 or (start_idx_0 + DEPTH) > SERIES_LEN: # Assertion
        print(f"FEHLER: Bild {IMAGE_IDX} kann mit Tiefe {DEPTH} nicht berechnet werden")
        sys.exit()

    X_raw, Y_raw = load_volume_by_start_index(H5_TEST_PATH, SERIES_IDX_0, start_idx_0, DEPTH)
    X_input = normalize(X_raw)

    model = models.load_model(MODEL_PATH, compile=False)

    input_shape = model.input_shape
    ndim = len(input_shape)

    if ndim == 4: 
        print(f"-> Erkannt: 2.5D Modell (Shape: {input_shape})")
        target_channels = input_shape[-1] # Hole Depth
        # Sanity check
        if target_channels != DEPTH:
            raise ValueError(f"Modell erwartet {target_channels} Channels, aber DEPTH ist {DEPTH}!")
        X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1))

    elif ndim == 5:
        print(f"-> Erkannt: 3D Modell (Shape: {input_shape})")
        X_feed = X_input # 3D Model nimmt Daten direkt
        
    else:
        raise ValueError(f"Unbekannter Input Shape: {input_shape}")

    Y_pred_raw = model.predict(X_feed, verbose=0) # Prediction
    
    # Immer die Mitte holen
    if Y_pred_raw.ndim == 5:
        mid_idx = Y_pred_raw.shape[1] // 2
        img_pred = Y_pred_raw[0, mid_idx, :, :, 0]
    elif Y_pred_raw.ndim == 4:
        img_pred = Y_pred_raw[0, :, :, 0]
    else:
        raise ValueError("Falsche output shape")

    # 
    image_low_count = X_input[0, center_offset, :, :, 0]
    image_ground_truth = normalize(Y_raw)[0, center_offset, :, :, 0]

    # Speichern
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    short_name = Path(MODEL_FILE).stem.split("seed")[0].rstrip("_")
    outfile = OUT_DIR / f"Pred_{short_name}_D{DEPTH}_S{SERIES_IDX}_Img{IMAGE_IDX}.npz"
    
    np.savez_compressed(outfile, lc=image_low_count, pred=img_pred, gt=image_ground_truth) # Speichern
    print(f"Fertig! Gespeichert als: {outfile.name}")

if __name__ == "__main__":
    main()