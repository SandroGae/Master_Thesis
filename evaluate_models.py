import os
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# GPU Optimierung
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# --- PFADE ---
HOME = Path.home()
DATA_ROOT = HOME / "data"
MODEL_DIR = DATA_ROOT / "checkpoints_unet_3d_simple"
TEST_DATA_DIR = DATA_ROOT / "original_data"
RESULT_FILE = HOME / "code/Master_Thesis/evaluation_results.txt"

# Deine Liste der Modelle
MODELS = [
    "cross_val_unet_25d_SSIM_middle_improved_V2_fold1_20260115-112330_loss0.0527_val0.0598.keras",
    "cross_val_unet_25d_SSIM_middle_improved_V2_fold2_20260115-112330_loss0.0527_val0.0549.keras",
    "cross_val_unet_25d_SSIM_middle_improved_V2_fold3_20260115-112330_loss0.0524_val0.0578.keras",
    "cross_val_unet_25d_SSIM_middle_improved_V2_fold4_20260115-112330_loss0.0539_val0.0530.keras",
    "cross_val_unet_25d_SSIM_middle_improved_V2_fold5_20260115-112330_loss0.0530_val0.0574.keras",
    "random_seed_unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20260115-163356_loss0.0521_val0.0585.keras",
    "no_augmentation_unet_25d_SSIM_middle_improved_V2_fold1_20260116-130113_loss0.0543_val0.0582.keras",
    "cross_val_unet_25d_SSIM_middle_improved_V2_interpolated_fold1_20260116-142232__best.keras"
]

# Deine 3 Test-Datensets
TEST_SETS = [
    "test_data.hdf5",
    "test_every_second_image.hdf5",
    "test_every_third_image.hdf5"
]

SERIES_LEN = 41

def mae_ssim_2d(y_true, y_pred, alpha=0.6):
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

def prepare_volumes_prealloc(X, y, depth):
    """Optimiert mit korrekter Achsen-Transformation für 2.5D."""
    n_vols = len(X) - depth + 1
    # Ziel-Form: (Anzahl_Fenster, Höhe, Breite, Tiefe)
    X_res = np.empty((n_vols, 192, 240, depth), dtype=np.float32)
    y_res = np.empty((n_vols, 192, 240, 1), dtype=np.float32)
    mid = depth // 2
    
    for i in range(n_vols):
        window = X[i : i + depth]
        window = np.squeeze(window, axis=-1)
        X_res[i] = np.transpose(window, (1, 2, 0))
        y_res[i] = y[i + mid]
        
    return X_res, y_res

def main():
    with open(RESULT_FILE, "w") as f:
        f.write(f"EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*145 + "\n")
        f.write(f"{'Model Name':<50} | {'Dataset':<25} | {'Loss':<10} | {'MAE':<10} | {'MSE':<12} | {'SSIM':<10}\n")
        f.write("-" * 145 + "\n")

    for model_name in MODELS:
        model_path = MODEL_DIR / model_name
        if not model_path.exists(): continue
        
        model = tf.keras.models.load_model(model_path, compile=False)
        depth = model.input_shape[3]

        for set_name in TEST_SETS:
            data_path = TEST_DATA_DIR / set_name
            if not data_path.exists(): continue
            
            print(f"Eval: {model_name[:20]}... on {set_name}")
            X_raw, y_raw = load_test_data(data_path)
            X_test, y_test = prepare_volumes_prealloc(X_raw, y_raw, depth)

            # Normalisierung & Clipping
            X_norm = (X_test / (np.sum(X_test, axis=(1,2,3), keepdims=True) + 1e-12)) * 10000.0
            y_norm = (y_test / (np.sum(y_test, axis=(1,2,3), keepdims=True) + 1e-12)) * 10000.0

            preds = model.predict(X_norm, batch_size=32, verbose=0)
            preds = np.clip(preds, 0.0, 1.0)
            y_norm = np.clip(y_norm, 0.0, 1.0)

            # Metriken
            y_t = tf.convert_to_tensor(y_norm); y_p = tf.convert_to_tensor(preds)
            loss_v = mae_ssim_2d(y_t, y_p).numpy()
            mae = np.mean(np.abs(preds - y_norm))
            mse = np.mean(np.square(preds - y_norm))
            ssim = tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0)).numpy()

            res_line = f"{model_name[:50]:<50} | {set_name:<25} | {loss_v:<10.6f} | {mae:<10.6f} | {mse:<12.8f} | {ssim:<10.6f}\n"
            with open(RESULT_FILE, "a") as f_out: f_out.write(res_line)
            
            tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()