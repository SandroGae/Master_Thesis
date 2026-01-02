import os
import sys
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Pfade
BASE_DIR = Path.home() / "VS_MASTER_THESIS"
MODEL_DIR = BASE_DIR / "Plots" / "Unet" / "Keras"
TEST_DATA_DIR = BASE_DIR / "original_data"
RESULT_FILE = BASE_DIR / "evaluation_results.txt"

MODELS = [
    MODEL_DIR / "1_unet_25d_SSIM_middle_improved_V2_interpolated_D5_VarStride1-24__20251201-093031_loss0.0690_val0.05942.keras",
    MODEL_DIR / "2_unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras",
]

TEST_SETS = [
    TEST_DATA_DIR / "test_data.hdf5",
    TEST_DATA_DIR / "test_every_second_image.hdf5",
    TEST_DATA_DIR / "test_every_third_image.hdf5"
]

def check_sanity():
    """Prüft Pfade und Dimensionen vor dem Start."""
    print("=== SANITY CHECK LÄUFT ===")
    all_fine = True
    
    # 1. Pfade prüfen
    for p in MODELS + TEST_SETS:
        if not p.exists():
            print(f"[ERROR] Datei nicht gefunden: {p}")
            all_fine = False
        else:
            print(f"[OK] Pfad existiert: {p.name}")

    if not all_fine:
        sys.exit("Abbruch: Ein oder mehrere Pfade sind ungültig.")

    # 2. Modell-Shapes und Daten-Shapes prüfen
    try:
        # Beispielhaft erstes Modell und erste Daten laden für Check
        m = tf.keras.models.load_model(MODELS[0], compile=False)
        m_shape = m.input_shape # (None, H, W, D) oder (None, D, H, W, 1)
        
        with h5py.File(TEST_SETS[0], "r") as f:
            d_shape = f["low_count/data"].shape # (H, W, N)
            
        print(f"[INFO] Modell erwartet (H, W): ({m_shape[1]}, {m_shape[2]})")
        print(f"[INFO] Daten liefern (H, W): ({d_shape[0]}, {d_shape[1]})")
        
        # Räumliche Auflösung (H, W) muss passen
        # Bei 3D ist die Shape (None, D, H, W, 1), H und W also Index 2 und 3
        if len(m_shape) == 5: # 3D
            m_h, m_w = m_shape[2], m_shape[3]
        else: # 2.5D
            m_h, m_w = m_shape[1], m_shape[2]

        if m_h != d_shape[0] or m_w != d_shape[1]:
            print(f"[ERROR] Dimension Mismatch! Modell: {m_h}x{m_w}, Daten: {d_shape[0]}x{d_shape[1]}")
            all_fine = False
            
    except Exception as e:
        print(f"[ERROR] Fehler beim Sanity Check: {e}")
        all_fine = False

    if all_fine:
        print("=== SANITY CHECK ERFOLGREICH. STARTE EVALUATION... ===\n")
    else:
        sys.exit("Abbruch wegen Inkompatibilität.")

def load_test_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        low = f["low_count/data"][:]   
        high = f["high_count/data"][:] 
    low = np.moveaxis(low, -1, 0)[..., np.newaxis].astype(np.float32)
    high = np.moveaxis(high, -1, 0)[..., np.newaxis].astype(np.float32)
    return low, high

def prepare_volumes(X, y, depth):
    n_vols = len(X) - depth + 1
    X_vols, y_targets = [], []
    mid = depth // 2
    for i in range(n_vols):
        vol = X[i : i + depth].squeeze() 
        vol = np.transpose(vol, (1, 2, 0))
        X_vols.append(vol)
        y_targets.append(y[i + mid])
    return np.array(X_vols), np.array(y_targets)

def main():
    # Erst prüfen, dann rechnen
    check_sanity()

    with open(RESULT_FILE, "w") as f:
        f.write(f"EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

    for model_path in MODELS:
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape
        current_depth, is_3d = (input_shape[1], True) if len(input_shape) == 5 else (input_shape[3], False)

        for data_path in TEST_SETS:
            X_raw, y_raw = load_test_data(data_path)
            X_test, y_test = prepare_volumes(X_raw, y_raw, current_depth)

            # Normalisierung & Prediction
            X_test_norm = np.array([(v / (np.sum(v) + 1e-12)) * 10000.0 for v in X_test])
            y_test_norm = y_test / (np.sum(y_test, axis=(1, 2), keepdims=True) + 1e-12)

            if is_3d:
                X_test_norm = np.transpose(X_test_norm, (0, 3, 1, 2))[..., np.newaxis]

            preds = model.predict(X_test_norm, batch_size=16, verbose=0)
            preds = np.clip(preds, 0, 1)
            y_test_norm = np.clip(y_test_norm, 0, 1)

            # Metriken
            mae = np.mean(np.abs(preds - y_test_norm))
            mse = np.mean(np.square(preds - y_test_norm))
            psnr = tf.image.psnr(y_test_norm, preds, max_val=1.0).numpy().mean()
            ssim = tf.image.ssim(tf.convert_to_tensor(y_test_norm), 
                                 tf.convert_to_tensor(preds), max_val=1.0).numpy().mean()

            # Output
            res_str = f"MODEL: {model_path.name}\nDATA:  {data_path.name}\nMAE: {mae:.6f} | SSIM: {ssim:.6f} | PSNR: {psnr:.2f}\n{'-'*60}\n"
            with open(RESULT_FILE, "a") as f: f.write(res_str)
            print(f"Done: {model_path.name} on {data_path.name}")
            
            del X_raw, y_raw, X_test, y_test, X_test_norm, y_test_norm, preds

if __name__ == "__main__":
    main()