import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- GPU Optimierung ---
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# --- PFADE & SETUP ---
# Script-Verzeichnis (Evaluation_test_set)
base_path = Path(__file__).parent
output_dir = base_path / "CSV_Evaluation_test_set"
output_dir.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path.home() / "data"
MODEL_DIR = DATA_ROOT / "checkpoints_unet_3d_simple"
TEST_DATA_DIR = DATA_ROOT / "original_data"

# --- MODELL-LISTE ---
MODELS = [
    # Random Seed Tests
    "random_seed_unet_25d_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20260115-163356_loss0.0521_val0.0585.keras",
    "random_seed_unet_25d_improved_V2__seed43__bf64__D5__lossMAE_SSIM__20260115-173320_loss0.0516_val0.0587.keras",
    "random_seed_unet_25d_improved_V2__seed44__bf64__D5__lossMAE_SSIM__20260115-183246_loss0.0521_val0.0586.keras",
    "random_seed_unet_25d_improved_V2__seed45__bf64__D5__lossMAE_SSIM__20260115-193254_loss0.0517_val0.0584.keras",
    "random_seed_unet_25d_improved_V2__seed46__bf64__D5__lossMAE_SSIM__20260115-203232_loss0.0519_val0.0587.keras",

    # Standard Cross-Validation
    "cross_val_unet_25d_improved_V2_fold1_20260115-112330_loss0.0527_val0.0598.keras",
    "cross_val_unet_25d_improved_V2_fold2_20260115-112330_loss0.0527_val0.0549.keras",
    "cross_val_unet_25d_improved_V2_fold3_20260115-112330_loss0.0524_val0.0578.keras",
    "cross_val_unet_25d_improved_V2_fold4_20260115-112330_loss0.0539_val0.0530.keras",
    "cross_val_unet_25d_improved_V2_fold5_20260115-112330_loss0.0530_val0.0574.keras",

    # No Augmentation Tests
    "no_augmentation_unet_25d_improved_V2_fold1_20260116-130113_loss0.0543_val0.0582.keras",
    "no_augmentation_unet_25d_improved_V2_fold2_20260116-130113_loss0.0535_val0.0560.keras",
    "no_augmentation_unet_25d_improved_V2_fold3_20260116-130113_loss0.0540_val0.0582.keras",
    "no_augmentation_unet_25d_improved_V2_fold4_20260116-130113_loss0.0535_val0.0532.keras",
    "no_augmentation_unet_25d_improved_V2_fold5_20260116-130113_loss0.0543_val0.0561.keras",

    # Interpolated Data Tests
    "cross_val_unet_25d_improved_V2_interpolated_fold1_20260116-142232_loss0.0684_val0.0573.keras",
    "fold2_only_unet_25d_improved_V2_interpolated_fold2_SEED42_20260119-105727_loss0.0703_val0.0533.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold3_20260116-142232_loss0.0684_val0.0561.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold4_20260116-142232_loss0.0679_val0.0582.keras",
    "cross_val_unet_25d_improved_V2_interpolated_fold5_20260116-142232_loss0.0702_val0.0558.keras",

    # No augmentation + Interpolated
    "no_augmentation_unet_25d_improved_V2_interpolated_fold1_20260119-104144_loss0.0703_val0.0574.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold2_20260119-104144_loss0.0704_val0.0533.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold3_20260119-104144_loss0.0717_val0.0566.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold4_20260119-104144_loss0.0687_val0.0589.keras",
    "no_augmentation_unet_25d_improved_V2_interpolated_fold5_20260119-104144_loss0.0716_val0.0546.keras",
]

TEST_SETS = ["test_data.hdf5", "test_every_second_image.hdf5", "test_every_third_image.hdf5"]

# --- HILFSFUNKTIONEN ---
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

# --- MAIN WORKFLOW ---
def main():
    raw_results = []
    avg_results = []

    for set_name in TEST_SETS:
        data_path = TEST_DATA_DIR / set_name
        if not data_path.exists(): continue
            
        print(f"\nProcessing Dataset: {set_name}")
        X_raw, y_raw = load_test_data(data_path)
        
        # Gruppenweise Verarbeitung (5 Modelle pro Gruppe)
        for i in range(0, len(MODELS), 5):
            model_group = MODELS[i : i + 5]
            group_metrics_store = {"Loss": [], "MAE": [], "MSE": [], "SSIM": [], "PSNR": []}
            
            # Gruppenname für die Auswertung bestimmen (Logik vom Parser)
            current_group = model_group[0].split('_fold')[0].split('__seed')[0]
            if "fold2_only" in current_group: # Edge Case Fix
                 current_group = "cross_val_unet_25d_improved_V2_interpolated"

            for model_name in model_group:
                model_path = MODEL_DIR / model_name
                if not model_path.exists(): 
                    print(f"Skipping: {model_name} (Not found)")
                    continue
                
                print(f"  Evaluating: {model_name}")
                model = tf.keras.models.load_model(model_path, compile=False)
                depth = model.input_shape[3]
                X_test, y_test = prepare_volumes_prealloc(X_raw, y_raw, depth)
                
                # Normalisierung
                X_norm = (X_test / (np.sum(X_test, axis=(1, 2), keepdims=True) + 1e-12)) * 10000.0
                y_norm = (y_test / (np.sum(y_test, axis=(1, 2), keepdims=True) + 1e-12)) * 10000.0

                preds = model.predict(X_norm, batch_size=32, verbose=0)
                preds = np.clip(preds, 0.0, 1.0)
                y_norm = np.clip(y_norm, 0.0, 1.0)

                # Per-Sample Metriken
                y_t = tf.convert_to_tensor(y_norm)
                y_p = tf.convert_to_tensor(preds)
                mse_per_sample = np.mean(np.square(preds - y_norm), axis=(1, 2, 3))
                
                loss_v = float(mae_ssim_2d(y_t, y_p).numpy())
                mae_v  = float(np.mean(np.abs(preds - y_norm)))
                mse_v  = float(np.mean(mse_per_sample))
                ssim_v = float(np.mean(tf.image.ssim(y_t, y_p, max_val=1.0).numpy()))
                psnr_v = float(np.mean(10.0 * np.log10(1.0 / (mse_per_sample + 1e-12))))

                # In Raw-Liste speichern
                raw_results.append({
                    'Dataset': set_name, 'Group': current_group, 'Model_Name': model_name,
                    'Loss': loss_v, 'MAE': mae_v, 'MSE': mse_v, 'SSIM': ssim_v, 'PSNR': psnr_v
                })

                # Für Gruppen-Durchschnitt sammeln
                group_metrics_store["Loss"].append(loss_v)
                group_metrics_store["MAE"].append(mae_v)
                group_metrics_store["MSE"].append(mse_v)
                group_metrics_store["SSIM"].append(ssim_v)
                group_metrics_store["PSNR"].append(psnr_v)

                tf.keras.backend.clear_session()

            # Durchschnitt der Gruppe berechnen
            if group_metrics_store["Loss"]:
                row_avg = {'Dataset': set_name, 'Group': current_group}
                for metric in ["Loss", "MAE", "MSE", "SSIM", "PSNR"]:
                    row_avg[f'{metric}_mean'] = np.mean(group_metrics_store[metric])
                    row_avg[f'{metric}_std'] = np.std(group_metrics_store[metric])
                avg_results.append(row_avg)

    # --- DATAFRAMES ERSTELLEN & SORTIEREN ---
    df_raw = pd.DataFrame(raw_results)
    df_avg = pd.DataFrame(avg_results)

    order = [
        'random_seed_unet_25d_improved_V2',
        'cross_val_unet_25d_improved_V2',
        'no_augmentation_unet_25d_improved_V2',
        'cross_val_unet_25d_improved_V2_interpolated',
        'no_augmentation_unet_25d_improved_V2_interpolated'
    ]
    
    existing_groups = [g for g in order if g in df_raw['Group'].unique()]
    df_raw['Group'] = pd.Categorical(df_raw['Group'], categories=existing_groups, ordered=True)
    df_avg['Group'] = pd.Categorical(df_avg['Group'], categories=existing_groups, ordered=True)
    
    df_raw = df_raw.sort_values(['Dataset', 'Group'])
    df_avg = df_avg.sort_values(['Dataset', 'Group'])

    # --- SPEICHERN ---
    df_raw.to_csv(output_dir / 'evaluation_raw.csv', index=False)
    df_avg.to_csv(output_dir / 'evaluation_averages.csv', index=False)
    
    print(f"\nFINITO! CSVs wurden direkt in {output_dir} gespeichert.")

if __name__ == "__main__":
    main()