#!/usr/bin/env python3
import numpy as np
import h5py
from tensorflow.keras import models
from pathlib import Path

# Konfiguration
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_DIR = ROOT_DIR / "Unet" / "Analysis_ROI" / "KERAS_MODEL"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw"

# Modelle laden
"""
MODELS = {
    "FILE_25D_3STACK": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras",
    "FILE_25D_5STACK": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras",
    "FILE_3d_middle_improved": "unet_3d_SSIM_middle_improved__seed42__bf64__D3__lossMAE_SSIM__20251112-180318_loss0.0479_val0.0522.keras",
    "FILE_3d_middle_improved_V2": "unet_3d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-144919_loss0.0479_val0.0517.keras",
    "FILE_25d_middle_improved_V2_kernel_3x5": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-103044_loss0.0523_val0.0587.keras",
    "FILE_25d_middle_improved_V2_interpolated": "unet_25d_D5_VarStride1-24__20251201-093031_loss0.0690_val0.0594.keras"
}
"""
"""
MODELS = {
    "Rang_1": "Rang_1_unet_25d_TripleLoss_a0.33_b0.17_bf64_D5_20260121-090819_loss0.0518_val0.0510.keras",
    "Rang_2": "Rang_2_unet_25d_TripleLoss_a0.17_b0.0_bf64_D5_20260121-012804_loss0.0259_val0.0296.keras",
    "Rang_3": "Rang_3_unet_25d_TripleLoss_a0.33_b0.33_bf64_D5_20260121-100752_loss0.0626_val0.0610.keras",
    "Rang_4": "Rang_4_unet_25d_TripleLoss_RESCUE_a0.17_b0.17_bf64_D5_20260123-223753_loss0.0450_val0.0428.keras",
    "Rang_5": "Rang_5_unet_25d_TripleLoss_RESCUE_a0.33_b0.0_bf64_D5_20260123-233434_loss0.0356_val0.0404.keras",
    "Rang_6": "Rang_6_unet_25d_TripleLoss_a0.17_b0.67_bf64_D5_20260121-051812_loss0.0981_val0.0815.keras",
    "Rang_7": "Rang_7_unet_25d_DeepScan_a0.25_b0.0833_bf64_D5_20260127-093131_loss0.0383_val0.0410.keras",
    "Rang_8": "Rang_8_unet_25d_DeepScan_a0.25_b0.0_bf64_D5_20260126-122819_loss0.0304_val0.0353.keras",
    "Rang_9": "Rang_9_unet_25d_TripleLoss_RESCUE_a0.17_b0.5_bf64_D5_20260123-214154_loss0.0832_val0.0685.keras",
    "Rang_10": "Rang_10_unet_25d_DeepScan_a0.25_b0.1667_bf64_D5_20260126-094704_loss0.0461_val0.0468.keras",
}
"""

MODELS = {
    # --- SUCCESS POINT 0 (P0) ---
    "P0_Seed43": "InfSeed_P0_a0.0000_b0.0000_seed43_20260210-170149_loss0.0195_val0.0224.keras",
    "P0_Seed44": "InfSeed_P0_a0.0000_b0.0000_seed44_20260210-180919_loss0.0195_val0.0224.keras",
    "P0_Seed47": "InfSeed_P0_a0.0000_b0.0000_seed47_20260210-193132_loss0.0158_val0.0181.keras",
    "P0_Seed50": "InfSeed_P0_a0.0000_b0.0000_seed50_20260210-204618_loss0.0191_val0.0219.keras",
    "P0_Seed62": "InfSeed_P0_a0.0000_b0.0000_seed62_20260211-001540_loss0.0152_val0.0180.keras",
    "P0_Seed63": "InfSeed_P0_a0.0000_b0.0000_seed63_20260211-013023_loss0.0155_val0.0180.keras",
    "P0_Seed65": "InfSeed_P0_a0.0000_b0.0000_seed65_20260211-024800_loss0.0154_val0.0182.keras",
    "P0_Seed69": "InfSeed_P0_a0.0000_b0.0000_seed69_20260212-092553_loss0.0161_val0.0182.keras",
    "P0_Seed75": "InfSeed_P0_a0.0000_b0.0000_seed75_20260212-110809_loss0.0203_val0.0226.keras",

    # --- SUCCESS POINT 1 (P1) ---
    "P1_Seed43": "InfSeed_P1_a0.8333_b0.0000_seed43_20260210-170150_loss0.0662_val0.0747.keras",
    "P1_Seed44": "InfSeed_P1_a0.8333_b0.0000_seed44_20260210-180249_loss0.0655_val0.0741.keras",
    "P1_Seed45": "InfSeed_P1_a0.8333_b0.0000_seed45_20260210-190307_loss0.0655_val0.0746.keras",
    "P1_Seed46": "InfSeed_P1_a0.8333_b0.0000_seed46_20260210-200941_loss0.0659_val0.0745.keras",
    "P1_Seed47": "InfSeed_P1_a0.8333_b0.0000_seed47_20260210-211638_loss0.0662_val0.0744.keras",
    "P1_Seed48": "InfSeed_P1_a0.8333_b0.0000_seed48_20260210-222216_loss0.0658_val0.0751.keras",
    "P1_Seed49": "InfSeed_P1_a0.8333_b0.0000_seed49_20260210-233303_loss0.0661_val0.0744.keras",
    "P1_Seed50": "InfSeed_P1_a0.8333_b0.0000_seed50_20260211-003034_loss0.0663_val0.0752.keras",
    "P1_Seed53": "InfSeed_P1_a0.8333_b0.0000_seed53_20260211-020101_loss0.0658_val0.0742.keras",
}

SERIES_LIST = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50] # Serien auswählen
SERIES_LEN = 41


# Hilfsfunktionen
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
    # Äußere Schleife: Modelle
    for chosen_name, model_file in MODELS.items():
        print(f"\n--- Suche Modell: {chosen_name} ---")
        
        # NEU: Sucht rekursiv in KERAS_MODEL und allen Unterordnern nach der Datei
        search_results = list(MODEL_DIR.rglob(model_file))
        
        if not search_results:
            print(f"Fehler: Die Datei '{model_file}' wurde in {MODEL_DIR} oder seinen Unterordnern nicht gefunden!")
            continue
        
        # Nimm den ersten Treffer
        model_path = search_results[0]
        print(f"Gefunden in: {model_path.parent.name}")

        model = models.load_model(model_path, compile=False)
        # ... Rest des Codes bleibt identisch ...
        input_dimension = len(model.input_shape)

        # Tiefe extrahieren
        depth = 5
        if "_D7_" in model_file: depth = 7
        elif "_D3_" in model_file: depth = 3

        # Innere Schleife: Serien
        for series_idx in SERIES_LIST:
            series_idx_0 = series_idx - 1
            print(f"Berechne Serie {series_idx}...")
            
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

                # Feed-Format anpassen
                if input_dimension == 4:
                    X_feed = np.transpose(np.squeeze(X_input, axis=-1), (0, 2, 3, 1)) 
                else:
                    X_feed = X_input

                # Prediction
                Y_pred_raw = model.predict(X_feed, verbose=0)
                
                # Slices extrahieren
                if Y_pred_raw.ndim == 5:
                    img_pred = Y_pred_raw[0, Y_pred_raw.shape[1]//2, :, :, 0]
                else:
                    img_pred = Y_pred_raw[0, :, :, 0]
                
                full_stack_lc[img_idx_0]   = X_input[0, center_offset, :, :, 0]
                full_stack_pred[img_idx_0] = img_pred
                full_stack_gt[img_idx_0]   = normalize(Y_raw)[0, center_offset, :, :, 0]

            # Speichern für diese Kombination aus Modell und Serie
            outfile = OUT_DIR / f"Pred_{chosen_name}_D{depth}_S{series_idx}_FullSeries.npz"
            np.savez_compressed(outfile, lc=full_stack_lc, pred=full_stack_pred, gt=full_stack_gt)
            print(f"Gespeichert: {outfile.name}")

    print("\nFertig! Alle Kombinationen berechnet.")

if __name__ == "__main__":
    main()