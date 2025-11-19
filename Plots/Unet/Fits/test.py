#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import h5py
import matplotlib
# matplotlib.use("Agg") # Auskommentieren, wenn du es direkt sehen willst (z.B. in IDE/Jupyter), sonst "Agg" für File-Save
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# Koordinaten (Pixel)
ROI_X_START = 30
ROI_X_END   = 120
ROI_Y_START = 100
ROI_Y_END   = 120
# ===========================

# =====================================================
# Daten-Hilfsfunktionen (Identisch zu deinem Skript)
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
    # Hier KEIN Clip auf 1.0, damit wir die echten Peaks in den Profilen sehen!
    # Der Clip ist meist nur für die Visualisierung des 2D Bildes wichtig.
    return volume

def get_vis_image(image):
    """Funktion nur für die 2D-Anzeige (99.5% Clip)"""
    img = image.copy()
    vmin, vmax = np.percentile(img, [0.5, 99.5])
    return (np.clip(img, vmin, vmax) - vmin) / (vmax - vmin)

# =====================================================
# Analyse Funktion
# =====================================================
def plot_roi_analysis(img_lc, img_pred, img_gt, roi_coords, out_path):
    """
    Erstellt den 3x3 Plot: Bilder mit Box + Horizontal Profil + Vertikal Profil
    """
    x1, x2, y1, y2 = roi_coords
    
    # 1. Daten extrahieren (Slicing)
    # Formate sind (H, W). Wir schneiden aus.
    crop_lc   = img_lc[y1:y2, x1:x2]
    crop_pred = img_pred[y1:y2, x1:x2]
    crop_gt   = img_gt[y1:y2, x1:x2]

    # 2. Profile berechnen (Durchschnitt entlang der orthogonalen Achse)
    # Horizontal: Wir laufen X ab -> also Average über Y (axis=0)
    prof_h_lc   = np.mean(crop_lc, axis=0)
    prof_h_pred = np.mean(crop_pred, axis=0)
    prof_h_gt   = np.mean(crop_gt, axis=0)

    # Vertikal: Wir laufen Y ab -> also Average über X (axis=1)
    prof_v_lc   = np.mean(crop_lc, axis=1)
    prof_v_pred = np.mean(crop_pred, axis=1)
    prof_v_gt   = np.mean(crop_gt, axis=1)

    # Achsen für Plots
    x_axis = np.arange(x1, x2)
    y_axis = np.arange(y1, y2) # Achtung: Im Bild läuft Y von oben nach unten

    # ================= PLOTTING =================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=150)
    
    cols = ["Low Count Input", "Prediction (Denoised)", "Ground Truth (High Count)"]
    imgs = [img_lc, img_pred, img_gt]
    h_profs = [prof_h_lc, prof_h_pred, prof_h_gt]
    v_profs = [prof_v_lc, prof_v_pred, prof_v_gt]
    colors = ['gray', 'red', 'black'] # Farben für die Linien

    # Zeile 1: 2D Bilder mit ROI Box
    for i in range(3):
        ax = axes[0, i]
        # Für Anzeige normalisieren wir schön auf 0-1 (visuell)
        vis_img = get_vis_image(imgs[i])
        ax.imshow(vis_img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{cols[i]}\nVol {VOL_IDX} (Series {SERIES_IDX})")
        
        # Rote Box zeichnen
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_axis_off()

    # Zeile 2: Horizontale Profile
    # Y-Limits gleich halten für Vergleichbarkeit?
    # Wir nehmen das Max von GT/Pred als Limit, damit man LC Rauschen sieht aber Skala passt
    ymax_h = max(np.max(prof_h_pred), np.max(prof_h_gt)) * 1.1
    ymin_h = min(np.min(prof_h_pred), np.min(prof_h_gt)) * 0.9

    for i in range(3):
        ax = axes[1, i]
        ax.plot(x_axis, h_profs[i], color=colors[i], marker='o', markersize=2, linewidth=1, label='Intensity')
        
        # Um Paper-Look zu imitieren (Gaussian Fit wäre hier extra, aber wir machen smooth line)
        ax.set_title(f"Horizontal Cut (avg Y={y1}-{y2})")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        # Optional: Gleiche Skala für alle 3
        # ax.set_ylim(ymin_h, ymax_h) 

    # Zeile 3: Vertikale Profile
    ymax_v = max(np.max(prof_v_pred), np.max(prof_v_gt)) * 1.1
    
    for i in range(3):
        ax = axes[2, i]
        # Achtung: Y-Achse im Plot vs Bild-Koordinaten. 
        # Standard Plot: X ist Index (Pixel Y koordinate), Y ist Intensität
        ax.plot(y_axis, v_profs[i], color=colors[i], marker='o', markersize=2, linewidth=1)
        
        ax.set_title(f"Vertical Cut (avg X={x1}-{x2})")
        ax.set_xlabel("Pixel Y")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        # Optional: Gleiche Skala
        # ax.set_ylim(0, ymax_v)

    plt.tight_layout()
    
    # Speichern
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot gespeichert: {out_path}")
    # plt.show() # Einkommentieren zum Anzeigen

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
    
    # Indizes berechnen
    # Wir wollen Series IDX und darin Vol IDX
    global_idx = (SERIES_IDX * n_vols_per_series) + VOL_IDX
    
    if global_idx >= len(X_vols):
        raise ValueError(f"Index {global_idx} out of bounds. Max: {len(X_vols)}")

    print(f"Analysiere Global Index: {global_idx} (Series {SERIES_IDX}, Vol {VOL_IDX})")

    # Daten holen (Shape: 1, Depth, H, W, 1)
    X_sample = X_vols[global_idx:global_idx+1]
    y_sample = y_vols[global_idx:global_idx+1]

    # Normalisieren (Wichtig: Hier nicht clippen für die Analyse!)
    X_norm = normalize_like_validation(X_sample, scale=10000.0)
    y_norm = normalize_like_validation(y_sample, scale=10000.0)

    # Predict
    print("Prediction...")
    pred = model.predict(X_norm, verbose=0) # Shape (1, 1, H, W, 1)

    # Bilder extrahieren (für 2D slices nehmen wir die Mitte der Depth)
    center_d = DEPTH // 2
    
    # Shapes in (H, W) bringen
    img_lc   = X_norm[0, center_d, :, :, 0]
    img_gt   = y_norm[0, center_d, :, :, 0]
    img_pred = pred[0, 0, :, :, 0]

    # Plotting
    out_file = OUT_DIR / f"Analysis_S{SERIES_IDX}_V{VOL_IDX}_ROI_X{ROI_X_START}-{ROI_X_END}_Y{ROI_Y_START}-{ROI_Y_END}.png"
    
    plot_roi_analysis(
        img_lc, 
        img_pred, 
        img_gt, 
        roi_coords=(ROI_X_START, ROI_X_END, ROI_Y_START, ROI_Y_END),
        out_path=out_file
    )

if __name__ == "__main__":
    main()