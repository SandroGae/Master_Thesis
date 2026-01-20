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
    # Standard Cross-Validation
    "cross_val_unet_25d_improved_V2_fold1_20260115-112330_loss0.0527_val0.0598.keras",
    "cross_val_unet_25d_improved_V2_fold2_20260115-112330_loss0.0527_val0.0549.keras",
    "cross_val_unet_25d_improved_V2_fold3_20260115-112330_loss0.0524_val0.0578.keras",
    "cross_val_unet_25d_improved_V2_fold4_20260115-112330_loss0.0539_val0.0530.keras",
    "cross_val_unet_25d_improved_V2_fold5_20260115-112330_loss0.0530_val0.0574.keras",

    # Random Seed Tests
    "random_seed_unet_25d_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20260115-163356_loss0.0521_val0.0585.keras",
    "random_seed_unet_25d_improved_V2__seed43__bf64__D5__lossMAE_SSIM__20260115-173320_loss0.0516_val0.0587.keras",
    "random_seed_unet_25d_improved_V2__seed44__bf64__D5__lossMAE_SSIM__20260115-183246_loss0.0521_val0.0586.keras",
    "random_seed_unet_25d_improved_V2__seed45__bf64__D5__lossMAE_SSIM__20260115-193254_loss0.0517_val0.0584.keras",
    "random_seed_unet_25d_improved_V2__seed46__bf64__D5__lossMAE_SSIM__20260115-203232_loss0.0519_val0.0587.keras",

    # No Augmentation Tests
    "no_augmentation_unet_25d_improved_V2_fold1_20260116-130113_loss0.0543_val0.0582.keras",
    "no_augmentation_unet_25d_improved_V2_fold2_20260116-130113_loss0.0535_val0.0560.keras",
    "no_augmentation_unet_25d_improved_V2_fold3_20260116-130113_loss0.0540_val0.0582.keras",
    "no_augmentation_unet_25d_improved_V2_fold4_20260116-130113_loss0.0535_val0.0532.keras",
    "no_augmentation_unet_25d_improved_V2_fold5_20260116-130113_loss0.0543_val0.0561.keras",

    # Interpolated Data Tests (Folds 1-5)
    "cross_val_unet_25d_improved_V2_interpolated_fold1_20260116-142232_loss0.0684_val0.0573.keras",
    "fold2_only_unet_25d_improved_V2_interpolated_fold2_SEED42_20260119-105727_loss0.0703_val0.0533.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold3_20260116-142232_loss0.0684_val0.0561.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold4_20260116-142232_loss0.0679_val0.0582.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold5_20260116-142232_loss0.0702_val0.0558.keras",

    # No augmentation + Interpolated (Folds 1-5)
    "no_augmentation_unet_25d_improved_V2_interpolated_fold1_20260119-104144_loss0.0703_val0.0574.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold2_20260119-104144_loss0.0704_val0.0533.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold3_20260119-104144_loss0.0717_val0.0566.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold4_20260119-104144_loss0.0687_val0.0589.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold5_20260119-104144_loss0.0716_val0.0546.keras",
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
    # Header schreiben
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*160 + "\n")
        f.write(f"{'Dataset':<30} | {'Model Name':<60} | {'Loss':<12} | {'MAE':<12} | {'MSE':<12} | {'SSIM':<10}\n")
        f.write("-" * 160 + "\n")

    # Äußere Schleife: Datensets
    for set_name in TEST_SETS:
        data_path = TEST_DATA_DIR / set_name
        if not data_path.exists():
            print(f"Skipping {set_name}: File not found.")
            continue
            
        print(f"\nProcessing Dataset: {set_name}")
        X_raw, y_raw = load_test_data(data_path)
        
        # Wir gehen die Liste MODELS in 5er-Schritten durch
        for i in range(0, len(MODELS), 5):
            model_group = MODELS[i : i + 5]
            
            # Speicher für die Metriken dieser 5er-Gruppe
            group_metrics = {
                "loss": [], "mae": [], "mse": [], "ssim": []
            }

            # Innere Schleife: Die 5 Modelle der aktuellen Gruppe
            for model_name in model_group:
                model_path = MODEL_DIR / model_name
                if not model_path.exists():
                    print(f"Warning: Model {model_name} not found.")
                    continue
                
                # Modell laden
                model = tf.keras.models.load_model(model_path, compile=False)
                depth = model.input_shape[3]
                
                # Daten vorbereiten & evaluieren
                X_test, y_test = prepare_volumes_prealloc(X_raw, y_raw, depth)
                
                # Normalisierung
                X_norm = (X_test / (np.sum(X_test, axis=(1,2,3), keepdims=True) + 1e-12)) * 10000.0
                y_norm = (y_test / (np.sum(y_test, axis=(1,2,3), keepdims=True) + 1e-12)) * 10000.0

                preds = model.predict(X_norm, batch_size=32, verbose=0)
                preds = np.clip(preds, 0.0, 1.0)
                y_norm = np.clip(y_norm, 0.0, 1.0)

                # Metriken berechnen
                y_t = tf.convert_to_tensor(y_norm)
                y_p = tf.convert_to_tensor(preds)
                
                loss_v = float(mae_ssim_2d(y_t, y_p).numpy())
                mae_v = float(np.mean(np.abs(preds - y_norm)))
                mse_v = float(np.mean(np.square(preds - y_norm)))
                ssim_v = float(tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0)).numpy())

                # In Gruppen-Metriken speichern
                group_metrics["loss"].append(loss_v)
                group_metrics["mae"].append(mae_v)
                group_metrics["mse"].append(mse_v)
                group_metrics["ssim"].append(ssim_v)

                # Einzel-Ergebnis schreiben
                res_line = f"{set_name:<30} | {model_name[:60]:<60} | {loss_v:<12.6f} | {mae_v:<12.6f} | {mse_v:<12.8f} | {ssim_v:<10.6f}\n"
                with open(RESULT_FILE, "a", encoding="utf-8") as f_out:
                    f_out.write(res_line)

                tf.keras.backend.clear_session()

            # Nachdem die 5 Modelle durch sind: Durchschnitt für die Gruppe berechnen
            if group_metrics["loss"]: # Nur berechnen, wenn die Gruppe nicht leer war
                avg_loss, std_loss = np.mean(group_metrics["loss"]), np.std(group_metrics["loss"])
                avg_mae,  std_mae  = np.mean(group_metrics["mae"]),  np.std(group_metrics["mae"])
                avg_mse,  std_mse  = np.mean(group_metrics["mse"]),  np.std(group_metrics["mse"])
                avg_ssim, std_ssim = np.mean(group_metrics["ssim"]), np.std(group_metrics["ssim"])

                stat_line = (
                    f"{'-'*30:30} | {'AVERAGE OF GROUP':<60} | "
                    f"{avg_loss:.4f}±{std_loss:.4f} | {avg_mae:.4f}±{std_mae:.4f} | "
                    f"{avg_mse:.6f}±{std_mse:.6f} | {avg_ssim:.4f}±{std_ssim:.4f}\n"
                )
                with open(RESULT_FILE, "a", encoding="utf-8") as f_out:
                    f_out.write(stat_line)
                    f_out.write("-" * 160 + "\n")

        # Trenner nach jedem Datenset-Block
        with open(RESULT_FILE, "a", encoding="utf-8") as f_out:
            f_out.write("="*160 + "\n")

    print("Evaluation FINITO")

if __name__ == "__main__":
    main()