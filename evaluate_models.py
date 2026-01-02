import os
import sys
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Pfade ---
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

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return (1.0 - alpha) * mae + alpha * (1.0 - tf.reduce_mean(ssim_val))

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
    total_runs = len(MODELS) * len(TEST_SETS)
    print(f"Starte korrekte Evaluation von {total_runs} Kombinationen...")

    with open(RESULT_FILE, "w") as f:
        f.write(f"EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HINWEIS: Normalisierung auf Summe 10.000 (identisch zu Training/Val)\n")
        f.write("="*75 + "\n\n")

    pbar = tqdm(total=total_runs, desc="Gesamtfortschritt")

    for model_path in MODELS:
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape
        current_depth = input_shape[3]

        for data_path in TEST_SETS:
            X_raw, y_raw = load_test_data(data_path)
            X_test, y_test = prepare_volumes(X_raw, y_raw, current_depth)

            # Normalisierung Input
            sums = np.sum(X_test, axis=(1, 2), keepdims=True) + 1e-12
            X_test_norm = (X_test / sums) * 10000.0
            
            # --- FIX: Ground Truth muss auch auf 10.000 skaliert werden! ---
            y_sums = np.sum(y_test, axis=(1, 2), keepdims=True) + 1e-12
            y_test_norm = (y_test / y_sums) * 10000.0 # <--- SKALIERUNG HINZUGEFÜGT

            # Prädiktion
            print(f"\nPrädiktion: {model_path.name}")
            preds = model.predict(X_test_norm, batch_size=16, verbose=1)
            
            # Clipping (identisch zu den Metrik-Funktionen im Training)
            preds = np.clip(preds, 0.0, 1.0)
            y_test_norm = np.clip(y_test_norm, 0.0, 1.0)

            # Metriken (Loss muss auf Tensor-Format für mae_ssim_2d)
            current_loss = mae_ssim_2d(tf.convert_to_tensor(y_test_norm), 
                                       tf.convert_to_tensor(preds)).numpy()
            
            mae = np.mean(np.abs(preds - y_test_norm))
            mse = np.mean(np.square(preds - y_test_norm))
            psnr = tf.reduce_mean(tf.image.psnr(y_test_norm, preds, max_val=1.0)).numpy()
            ssim = tf.reduce_mean(tf.image.ssim(tf.convert_to_tensor(y_test_norm), 
                                               tf.convert_to_tensor(preds), max_val=1.0)).numpy()

            res_str = (
                f"MODEL: {model_path.name}\n"
                f"DATA:  {data_path.name}\n"
                f"LOSS:  {current_loss:.6f} (MAE_SSIM)\n"
                f"MAE:   {mae:.6f} | MSE: {mse:.8f} | SSIM: {ssim:.6f} | PSNR: {psnr:.2f} dB\n"
                f"{'-'*75}\n"
            )
            
            with open(RESULT_FILE, "a") as f_out:
                f_out.write(res_str)
            
            pbar.update(1)
            del X_raw, y_raw, X_test, y_test, X_test_norm, y_test_norm, preds

    pbar.close()
    print(f"\nEvaluation fertig. PSNR sollte jetzt wieder bei ~30-31 liegen.")

if __name__ == "__main__":
    main()