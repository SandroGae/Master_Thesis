#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import h5py
import matplotlib
# matplotlib.use("Agg") # Falls du keine GUI hast (Server), einkommentieren
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit

# =====================================================
# KONFIGURATION & ROI
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / "unet_3d_SSIM_middle__seed42__bf64__D3__lossMAE_SSIM__20251112-231304_loss0.0481_val0.0518.keras"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI"

# Parameter
DEPTH = 3
SERIES_LEN = 41
SERIES_IDX = 11     # Die Serie
VOL_IDX = 17        # Das 18. Bild (Index 17)

# === HIER ROI DEFINIEREN ===
# Wähle die ROI so, dass der Peak mittig ist und links/rechts Hintergrund
ROI_X_START = 30
ROI_X_END   = 120
ROI_Y_START = 100
ROI_Y_END   = 120
# ===========================

# =====================================================
# 1. Hilfsfunktionen: Daten laden & Vorbereiten
# =====================================================
def load_test_split(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:] 
        high_count = f["high_count/data"][:] 
    low_count = np.moveaxis(low_count, -1, 0)[..., np.newaxis]
    high_count = np.moveaxis(high_count, -1, 0)[..., np.newaxis]
    return low_count.astype(np.float32), high_count.astype(np.float32)

def make_sliding_windows(X, y, series_len, depth):
    N, H, W, C = X.shape
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1
    X_volumes, y_volumes = [], []
    for i in range(n_series):
        start = i * series_len
        blockX = X[start:start + series_len]
        blockY = y[start:start + series_len]
        for start_idx in range(n_vols_per_series):
            X_volumes.append(blockX[start_idx:start_idx + depth])
            y_volumes.append(blockY[start_idx:start_idx + depth])
    return np.stack(X_volumes, axis=0), np.stack(y_volumes, axis=0), n_series, n_vols_per_series

def normalize_like_validation(volumes, scale=10000.0):
    volume = np.maximum(volumes, 0.0).astype(np.float32)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    volume = volume / sums
    volume = volume * scale
    return volume

# =====================================================
# 2. Analyse-Funktionen (Paper Style)
# =====================================================
def gaussian(x, amp, mu, sigma):
    """Gauß-Kurve für den Fit"""
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian_and_remove_bg(y_data):
    """
    Simuliert die Analyse aus dem Paper:
    1. Zieht linearen Hintergrund ab (basierend auf Rändern).
    2. Fittet einen Gauss.
    """
    x = np.arange(len(y_data))
    
    # A) Hintergrund schätzen (Durchschnitt der ersten/letzten 3 Punkte)
    bg_start = np.mean(y_data[:3])
    bg_end   = np.mean(y_data[-3:])
    
    # Lineare Funktion für Hintergrund abziehen
    bg_line = np.linspace(bg_start, bg_end, len(y_data))
    y_corrected = y_data - bg_line
    
    # B) Gaussian Fit versuchen
    # Start-Schätzung: Amplitude=Max, Mu=Wo ist Max, Sigma=5
    p0 = [np.max(y_corrected), np.argmax(y_corrected), 5.0]
    
    try:
        # curve_fit findet die optimalen Parameter für die Gauss-Funktion
        popt, _ = curve_fit(gaussian, x, y_corrected, p0=p0, maxfev=5000)
        y_fit = gaussian(x, *popt)
    except Exception as e:
        print(f"Info: Kein Fit möglich (zu viel Rauschen oder kein Peak).")
        y_fit = None
        
    return x, y_corrected, y_fit

def plot_roi_analysis_paper_style(img_lc, img_pred, img_gt, roi_coords, out_path):
    x1, x2, y1, y2 = roi_coords
    
    # Crops ausschneiden
    crop_lc   = img_lc[y1:y2, x1:x2]
    crop_pred = img_pred[y1:y2, x1:x2]
    crop_gt   = img_gt[y1:y2, x1:x2]

    # Profile berechnen (SUMME über die jeweils andere Achse)
    # Horizontal (Y wird aufsummiert -> Verlauf über X)
    prof_h_lc   = np.sum(crop_lc, axis=0)
    prof_h_pred = np.sum(crop_pred, axis=0)
    prof_h_gt   = np.sum(crop_gt, axis=0)

    # Vertikal (X wird aufsummiert -> Verlauf über Y)
    prof_v_lc   = np.sum(crop_lc, axis=1)
    prof_v_pred = np.sum(crop_pred, axis=1)
    prof_v_gt   = np.sum(crop_gt, axis=1)
    
    # Setup für den Plot
    h_profs = [prof_h_lc, prof_h_pred, prof_h_gt]
    v_profs = [prof_v_lc, prof_v_pred, prof_v_gt]
    titles  = ["Input (LC)", "Denoised (CNN)", "Ground Truth (HC)"]
    imgs    = [img_lc, img_pred, img_gt]
    colors  = ['gray', 'red', 'black'] 
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=150)

    # --- REIHE 1: BILDER MIT BOX ---
    for i in range(3):
        ax = axes[0, i]
        # Visuelle Normalisierung (nur für das Bild, nicht die Daten)
        vis = imgs[i].copy()
        vmin, vmax = np.percentile(vis, [1, 99])
        # Vermeiden von Division durch Null bei flachen Bildern
        if vmax > vmin:
            vis = (np.clip(vis, vmin, vmax) - vmin) / (vmax - vmin)
        
        ax.imshow(vis, cmap="gray_r")
        # ROI Rechteck einzeichnen
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')

    # --- REIHE 2: HORIZONTALE CUTS (X-Achse) ---
    for i in range(3):
        ax = axes[1, i]
        
        # Hintergrund abziehen & Fitten
        x_axis, y_corr, y_fit = fit_gaussian_and_remove_bg(h_profs[i])
        
        # Punkte plotten
        ax.errorbar(x_axis, y_corr, fmt='o', color=colors[i], markersize=3, alpha=0.6, label='Data')
        
        # Fit Linie plotten (wenn Fit erfolgreich)
        if y_fit is not None:
            ax.plot(x_axis, y_fit, color=colors[i], linewidth=2, label='Gaussian Fit')
            
        ax.set_title("Horizontal Scan (BG subtracted)")
        ax.set_xlabel("Pixel X")
        if i == 0: ax.set_ylabel("Intensity (Sum)")
        ax.grid(True, alpha=0.3)
        if i == 1: ax.legend(loc='upper right') # Legende nur beim mittleren Plot nötig

    # --- REIHE 3: VERTIKALE CUTS (Y-Achse) ---
    for i in range(3):
        ax = axes[2, i]
        
        x_axis, y_corr, y_fit = fit_gaussian_and_remove_bg(v_profs[i])
        
        ax.errorbar(x_axis, y_corr, fmt='o', color=colors[i], markersize=3, alpha=0.6, label='Data')
        
        if y_fit is not None:
            ax.plot(x_axis, y_fit, color=colors[i], linewidth=2, label='Gaussian Fit')
            
        ax.set_title("Vertical Scan (BG subtracted)")
        ax.set_xlabel("Pixel Y")
        if i == 0: ax.set_ylabel("Intensity (Sum)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Paper-Style Plot gespeichert: {out_path}")

# =====================================================
# MAIN
# =====================================================
def main():
    print(f"Lade Modell: {MODEL_PATH.name}")
    model = models.load_model(MODEL_PATH, compile=False)

    print("Lade Daten...")
    X_test, y_test = load_test_split(H5_TEST_PATH)
    
    # Sliding Window
    X_vols, y_vols, n_series, n_vols_per_series = make_sliding_windows(X_test, y_test, SERIES_LEN, DEPTH)
    
    global_idx = (SERIES_IDX * n_vols_per_series) + VOL_IDX
    
    if global_idx >= len(X_vols):
        raise ValueError(f"Index {global_idx} out of bounds. Max: {len(X_vols)}")

    print(f"Analysiere Global Index: {global_idx} (Series {SERIES_IDX}, Vol {VOL_IDX})")

    # Daten holen & Normalisieren
    X_sample = X_vols[global_idx:global_idx+1]
    y_sample = y_vols[global_idx:global_idx+1]

    X_norm = normalize_like_validation(X_sample, scale=10000.0)
    y_norm = normalize_like_validation(y_sample, scale=10000.0)

    # Predict
    print("Prediction...")
    pred = model.predict(X_norm, verbose=0)

    # Bilder extrahieren (Mitte der Tiefe)
    center_d = DEPTH // 2
    img_lc   = X_norm[0, center_d, :, :, 0]
    img_gt   = y_norm[0, center_d, :, :, 0]
    img_pred = pred[0, 0, :, :, 0]

    # Plotting
    out_file = OUT_DIR / f"Analysis_S{SERIES_IDX}_V{VOL_IDX}_PaperStyle_ROI_X{ROI_X_START}-{ROI_X_END}.png"
    
    # HIER WAR DER FEHLER: Jetzt rufen wir die richtige Funktion auf!
    plot_roi_analysis_paper_style(
        img_lc, 
        img_pred, 
        img_gt, 
        roi_coords=(ROI_X_START, ROI_X_END, ROI_Y_START, ROI_Y_END),
        out_path=out_file
    )

if __name__ == "__main__":
    main()