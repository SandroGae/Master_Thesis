#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import h5py
from pathlib import Path
import pandas as pd
import os

# GPU Konfiguration (optional, verhindert OOM bei Inference großer Batches)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# ==========================================
# KONFIGURATION
# ==========================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"

# Deine Modell-Liste
MODELS_TO_EVAL = {
    "2.5D_Stack3_Avg":  "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras",
    "2.5D_Stack5_Std":  "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras",
    "2.5D_Stack5_Dilated": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-103044_loss0.0523_val0.0587.keras"
}

MODEL_DIR = ROOT_DIR / "Plots" / "Unet" / "Keras"

# Konstanten (müssen zum Training passen)
SERIES_LEN = 41
NORM_SCALE = 10000.0  # Validation Scale aus dem Training



def mae_scaled(y_true, y_pred):
    """
    MAE auf den Modell-Outputs (Wertebereich 0..1).
    Entspricht dem 'mae_center' aus dem Training.
    """
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    y_p = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.abs(y_t - y_p))

def mae_norm(y_true, y_pred):
    """
    MAE normalisiert auf Skala 1 (geteilt durch 10.000).
    Dies gibt den relativen Fehler bezogen auf den Skalierungsfaktor an.
    """
    return mae_scaled(y_true, y_pred) / NORM_SCALE

def psnr_scaled(y_true, y_pred):
    """PSNR auf den skalierten Daten (Peak ist 1.0 durch Sigmoid)"""
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    y_p = tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_t, y_p), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_scaled(y_true, y_pred):
    """SSIM auf den skalierten Daten"""
    y_t = tf.clip_by_value(y_true, 0.0, 1.0)
    y_p = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))

# ==========================================
# DATEN LADEN & PREPROCESSING
# ==========================================
def load_test_data(h5_path):
    print(f"Lade Rohdaten: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        low = f["low_count/data"][:]
        high = f["high_count/data"][:]
    
    # (H, W, N) -> (N, H, W, 1)
    low = np.moveaxis(low, -1, 0)[..., np.newaxis]
    high = np.moveaxis(high, -1, 0)[..., np.newaxis]
    
    return low.astype(np.float32), high.astype(np.float32)

def normalize_validation_style(volume, scale=NORM_SCALE):
    """
    1. Clip negative
    2. Slice-wise sum norm
    3. Scale multiplication
    """
    vol = np.maximum(volume, 0.0)
    # Summe über H, W, C
    sums = np.sum(vol, axis=(1, 2, 3), keepdims=True) + 1e-12
    vol = (vol / sums) * scale
    return vol

def create_windows(X, y, depth, series_len):
    """
    Wandelt (N, H, W, 1) -> Input (N_samples, H, W, Depth), Target (N_samples, H, W, 1)
    """
    N = X.shape[0]
    n_series = N // series_len
    n_per_series = series_len - depth + 1
    
    X_list = []
    y_list = []
    
    center_offset = depth // 2

    for s in range(n_series):
        base = s * series_len
        bx = X[base : base+series_len]
        by = y[base : base+series_len]
        
        for i in range(n_per_series):
            # Input Stack: (Depth, H, W, 1)
            stack = bx[i : i+depth] 
            
            # Transformation für 2.5D Input (Depth wird zu Channel):
            # Squeeze (Depth, H, W) -> Transpose (H, W, Depth)
            stack_sq = np.squeeze(stack, axis=-1)
            stack_tr = np.transpose(stack_sq, (1, 2, 0)) 
            
            X_list.append(stack_tr)
            
            # Target: Mittleres Slice
            y_mid = by[i + center_offset]
            y_list.append(y_mid)

    return np.array(X_list), np.array(y_list)

# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Daten laden
    if not H5_TEST_PATH.exists():
        raise FileNotFoundError(f"Testdaten nicht gefunden: {H5_TEST_PATH}")
        
    X_raw, y_raw = load_test_data(H5_TEST_PATH)
    
    print("Normalisiere Daten (Validation-Style)...")
    X_norm = normalize_validation_style(X_raw)
    y_norm = normalize_validation_style(y_raw)
    
    results_list = []

    # 2. Modelle evaluieren
    for model_name, fname in MODELS_TO_EVAL.items():
        fpath = MODEL_DIR / fname
        if not fpath.exists():
            print(f"SKIP: {model_name} (Datei nicht gefunden: {fname})")
            continue
            
        print(f"\n--- Evaluiere: {model_name} ---")
        
        # A. Modell laden (compile=False ist sicherer bei Custom Layers)
        model = models.load_model(fpath, compile=False)
        
        # B. Tiefe erkennen
        try:
            inp_shape = model.input_shape
            # Keras kann input_shape als Tuple oder Liste von Tuples zurückgeben
            if isinstance(inp_shape, list): 
                inp_shape = inp_shape[0]
            depth = inp_shape[-1]
            print(f" -> Erkannte Tiefe (Channels): {depth}")
        except Exception as e:
            depth = 5
            print(f" -> Tiefe nicht erkannt, Fallback auf {depth}. Fehler: {e}")

        # C. Sliding Windows erstellen
        print(f" -> Erstelle Windows (Depth={depth})...")
        X_test, y_test = create_windows(X_norm, y_norm, depth, SERIES_LEN)
        
        # D. Manuelles Compilieren für Metriken
        # Wir benutzen die korrigierten Metriken mit Clipping
        metrics_list = [mae_scaled, mae_norm, psnr_scaled, ssim_scaled]
        model.compile(loss='mae', metrics=metrics_list)
        
        # E. Evaluieren
        print(" -> Berechne Metriken...")
        scores = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
        
        # Ergebnis speichern
        res_dict = {"Model": model_name, "Depth": depth}
        for name, val in zip(model.metrics_names, scores):
            res_dict[name] = val
        results_list.append(res_dict)

    # 3. Output
    print("\n\n================ ZUSAMMENFASSUNG ================")
    if not results_list:
        print("Keine Modelle evaluiert.")
        return

    df = pd.DataFrame(results_list)
    
    # Sortierung und Spaltenauswahl für Übersichtlichkeit
    desired_cols = ["Model", "Depth", "mae_scaled", "psnr_scaled", "ssim_scaled", "mae_norm"]
    final_cols = [c for c in desired_cols if c in df.columns]
    
    # Schönere Formatierung
    pd.options.display.float_format = '{:,.5f}'.format
    print(df[final_cols].to_string(index=False))
    
    out_csv = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Testset_Comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nErgebnisse gespeichert in: {out_csv}")

if __name__ == "__main__":
    main()