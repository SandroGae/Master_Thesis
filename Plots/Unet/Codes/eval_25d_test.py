#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import h5py
from pathlib import Path
import pandas as pd # Für schöne Tabellen am Ende

# ==========================================
# KONFIGURATION
# ==========================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
H5_TEST_PATH = ROOT_DIR / "data" / "original_data" / "test_data.hdf5"

# Liste deiner Modelle (Name: Dateiname)
MODELS_TO_EVAL = {
    "2.5D_Stack3_Avg":  "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-181534_loss0.0535_val0.0597.keras",
    "2.5D_Stack5_Std":  "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251119-171216_loss0.0519_val0.0585.keras",
    "2.5D_Stack5_Dilated": "unet_25d_SSIM_middle_improved_V2__seed42__bf64__D5__lossMAE_SSIM__20251201-103044_loss0.0523_val0.0587.keras"
}

MODEL_DIR = ROOT_DIR / "Plots" / "Unet" / "Keras"

# Konstanten
SERIES_LEN = 41
NORM_SCALE = 10000.0  # Validation Scale

# ==========================================
# METRIKEN (Wie besprochen: Normalized & Raw)
# ==========================================
def mae_norm(y_true, y_pred):
    """MAE auf 0..1 normalisiert (Vergleichbar mit alten Runs)"""
    return tf.reduce_mean(tf.abs(y_true - y_pred)) / NORM_SCALE

def psnr_norm(y_true, y_pred):
    """PSNR als ob max=1.0 wäre"""
    y_t = tf.clip_by_value(y_true / NORM_SCALE, 0.0, 1.0)
    y_p = tf.clip_by_value(y_pred / NORM_SCALE, 0.0, 1.0)
    mse = tf.reduce_mean(tf.math.squared_difference(y_t, y_p), axis=(1,2,3))
    return 10.0 * tf.math.log(1.0 / (mse + 1e-12)) / tf.math.log(10.0)

def ssim_norm(y_true, y_pred):
    """SSIM auf 0..1 Skala"""
    y_t = tf.clip_by_value(y_true / NORM_SCALE, 0.0, 1.0)
    y_p = tf.clip_by_value(y_pred / NORM_SCALE, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_t, y_p, max_val=1.0))

def mae_raw(y_true, y_pred):
    """Echter Fehler in Photonen"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# ==========================================
# DATEN LADEN & PREPROCESSING
# ==========================================
def load_test_data(h5_path):
    print(f"Lade Rohdaten: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        # Shape: (Height, Width, N_Total) -> wir machen (N, H, W, 1) daraus
        low = f["low_count/data"][:]
        high = f["high_count/data"][:]
    
    # Move Axis: (H, W, N) -> (N, H, W)
    low = np.moveaxis(low, -1, 0)
    high = np.moveaxis(high, -1, 0)
    
    # Add Channel: (N, H, W, 1)
    low = low[..., np.newaxis]
    high = high[..., np.newaxis]
    
    return low.astype(np.float32), high.astype(np.float32)

def normalize_validation_style(volume, scale=NORM_SCALE):
    """
    Exakt deine Validierungs-Logik:
    1. Clip negative Werte
    2. Summe über (H,W,C) -> 1
    3. Mal 10.000
    """
    vol = np.maximum(volume, 0.0)
    sums = np.sum(vol, axis=(1, 2, 3), keepdims=True) + 1e-12
    vol = (vol / sums) * scale
    return vol

def create_windows(X, y, depth, series_len):
    """
    Erstellt (N_samples, H, W, Depth) für Input
    und (N_samples, H, W, 1) für Output (Mitte)
    """
    N = X.shape[0]
    n_series = N // series_len
    n_per_series = series_len - depth + 1
    
    X_list = []
    y_list = []
    
    center_offset = depth // 2

    for s in range(n_series):
        base = s * series_len
        # Slice den Block für diese Serie
        bx = X[base : base+series_len]
        by = y[base : base+series_len]
        
        for i in range(n_per_series):
            # Input: Stack der Tiefe D
            # Shape bx[i:i+depth]: (Depth, H, W, 1)
            stack = bx[i : i+depth] 
            
            # Wir müssen das umformen zu (H, W, Depth) für 2D Conv
            # Aktuell: (Depth, H, W, 1) -> squeeze -> (Depth, H, W) -> transpose -> (H, W, Depth)
            stack_sq = np.squeeze(stack, axis=-1)
            stack_tr = np.transpose(stack_sq, (1, 2, 0)) # H, W, D
            
            X_list.append(stack_tr)
            
            # Output: Das mittlere Bild
            # Shape: (H, W, 1)
            y_mid = by[i + center_offset]
            y_list.append(y_mid)

    return np.array(X_list), np.array(y_list)

# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Daten laden & Normalisieren (nur einmal nötig)
    X_raw, y_raw = load_test_data(H5_TEST_PATH)
    
    print("Normalisiere Daten (Validation-Style auf 10k)...")
    X_norm = normalize_validation_style(X_raw)
    y_norm = normalize_validation_style(y_raw)
    
    results_list = []

    # 2. Schleife über alle Modelle
    for model_name, fname in MODELS_TO_EVAL.items():
        fpath = MODEL_DIR / fname
        if not fpath.exists():
            print(f"SKIP: {model_name} (Datei nicht gefunden)")
            continue
            
        print(f"\n--- Evaluiere: {model_name} ---")
        
        # A. Modell laden
        # compile=False, weil wir gleich manuell compilen für Metriken
        model = models.load_model(fpath, compile=False)
        
        # B. Tiefe automatisch erkennen
        try:
            inp = model.input_shape
            if isinstance(inp, list): inp = inp[0]
            depth = inp[-1] # Letzte Dimension ist Channel/Depth bei (H, W, D)
            print(f" -> Erkannte Tiefe: {depth}")
        except:
            depth = 5
            print(f" -> Tiefe nicht erkannt, rate {depth}")

        # C. Daten für diese Tiefe vorbereiten
        print(" -> Erstelle Sliding Windows...")
        X_test, y_test = create_windows(X_norm, y_norm, depth, SERIES_LEN)
        print(f" -> Input Shape: {X_test.shape}, Target Shape: {y_test.shape}")
        
        # D. Compilieren mit unseren Metriken
        # Wir nutzen hier eine Dummy-Loss Funktion oder einfach MAE, 
        # da wir nur model.evaluate machen.
        model.compile(loss='mae', metrics=[mae_norm, psnr_norm, ssim_norm, mae_raw])
        
        # E. Evaluieren
        print(" -> Berechne Metriken...")
        scores = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
        
        # Scores ist eine Liste: [loss, metric1, metric2, ...]
        # Keras gibt uns ein Dictionary map zurück in neueren Versionen oder wir mappen es manuell
        res_dict = {"Model": model_name, "Depth": depth}
        # model.metrics_names enthält die Namen passend zu scores
        for name, val in zip(model.metrics_names, scores):
            res_dict[name] = val
            
        results_list.append(res_dict)

    # 3. Ergebnis ausgeben
    print("\n\n================ ZUSAMMENFASSUNG ================")
    df = pd.DataFrame(results_list)
    
    # Spalten schön anordnen falls vorhanden
    cols = ["Model", "Depth", "mae_norm", "psnr_norm", "ssim_norm", "mae_raw"]
    # Nur Spalten nutzen, die auch wirklich da sind
    final_cols = [c for c in cols if c in df.columns]
    
    print(df[final_cols].to_string(index=False))
    
    # Optional: Als CSV speichern
    out_csv = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Metric_Comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nGespeichert in: {out_csv}")

if __name__ == "__main__":
    main()