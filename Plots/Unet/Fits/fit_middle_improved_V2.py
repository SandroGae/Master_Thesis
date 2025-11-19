#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit   # ggf. conda install scipy

# =====================================================
# Pfade
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
MODEL_PATH = ROOT_DIR / "Plots" / "Unet" / "Keras" / "unet_3d_SSIM_middle__seed42__bf64__D3__lossMAE_SSIM__20251112-231304_loss0.0481_val0.0518.keras"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR = ROOT_DIR / "Plots" / "Unet" / "Gauss_Fits"

CLIP = False
DEPTH = 3
SERIES_LEN = 41
N_VOLS_PER_SERIES = SERIES_LEN - DEPTH + 1  # = 39

# =========================================
# Welches Volumen?
# =========================================
SERIES_IDX = 11   # Serie (0-basiert)
TARGET_VOL = 17   # Volumen innerhalb dieser Serie (0-basiert) -> Bild 18

# =========================================
# ROI-Parameter (relativ zur Bildgroesse)
# =========================================
ROI_Y_CENTER_FRAC = 0.60  # vertikale Position der ROI-Mitte (0..1, 0=oben)
ROI_HEIGHT_FRAC   = 0.10  # relative Hoehe der ROI
ROI_X_START_FRAC  = 0.10  # Start in x-Richtung
ROI_X_END_FRAC    = 0.90  # Ende in x-Richtung

# =========================================
# Subsampling fuer die Fits
# jedes STEP_PROFILE-te Pixel wird verwendet
# =========================================
STEP_PROFILE = 3   # z.B. 1 = jedes Pixel, 3 = jedes dritte Pixel

# =========================================
# Achsen-Limits (None = automatisch)
# =========================================
# horizontale Profile (x-Achse = Pixel in ROI, y-Achse = Intensitaet)
H_X_LIM = None          # z.B. (0, 250) oder None
H_Y_LIM = (0.0, 1.0)    # z.B. (0.0, 1.0) oder None

# vertikale Profile (x-Achse = Pixel in ROI, y-Achse = Intensitaet)
V_X_LIM = None
V_Y_LIM = (0.0, 1.0)

# =====================================================
# Daten-Hilfsfunktionen
# =====================================================
def load_test_split(h5_path: Path):
    """
    Laedt Testdaten aus HDF5-Datei und formatiert sie:
    (H, W, N) -> (N, H, W, 1)
    """
    with h5py.File(h5_path, "r") as f:
        low_count = f["low_count/data"][:]   # (H, W, N)
        high_count = f["high_count/data"][:] # (H, W, N)

    low_count = np.moveaxis(low_count, -1, 0)   # (N, H, W)
    high_count = np.moveaxis(high_count, -1, 0) # (N, H, W)

    low_count = low_count[..., np.newaxis]      # (N, H, W, 1)
    high_count = high_count[..., np.newaxis]    # (N, H, W, 1)

    return low_count.astype(np.float32), high_count.astype(np.float32)


def make_sliding_windows(X, y, series_len, depth):
    """
    X, y: (N, H, W, 1), N = num_series * series_len
    Output:
        X_vols, y_vols: (N_vols_total, D, H, W, 1)
        n_series, n_vols_per_series
    """
    N, H, W, C = X.shape
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    n_series = N // series_len
    n_vols_per_series = series_len - depth + 1

    X_volumes = []
    y_volumes = []

    for i in range(n_series):
        start = i * series_len
        blockX = X[start:start + series_len]
        blockY = y[start:start + series_len]

        for start_idx in range(n_vols_per_series):
            X_volumes.append(blockX[start_idx:start_idx + depth])
            y_volumes.append(blockY[start_idx:start_idx + depth])

    X_volumes = np.stack(X_volumes, axis=0)
    y_volumes = np.stack(y_volumes, axis=0)
    return X_volumes, y_volumes, n_series, n_vols_per_series


def normalize_like_validation(volumes, scale=10000.0, do_clip=CLIP):
    """
    Normierung wie im Validation-Set:
    - clip >= 0
    - pro Slice durch Summe ueber (H,W,C) teilen
    - mit 'scale' multiplizieren
    volumes: (N, D, H, W, 1)
    """
    volume = np.maximum(volumes, 0.0).astype(np.float32)
    sums = np.sum(volume, axis=(2, 3, 4), keepdims=True) + 1e-12
    volume = volume / sums
    volume = volume * scale
    if do_clip:
        volume = np.clip(volume, 0.0, 1.0)
    return volume

# =====================================================
# ROI, Profile, Gauss-Fit
# =====================================================
def get_roi_indices(H, W):
    """Berechnet Pixelgrenzen der ROI aus den relativen Parametern."""
    roi_height = int(ROI_HEIGHT_FRAC * H)
    y_center = int(ROI_Y_CENTER_FRAC * H)
    y0 = max(0, y_center - roi_height // 2)
    y1 = min(H, y_center + roi_height // 2)

    x0 = int(ROI_X_START_FRAC * W)
    x1 = int(ROI_X_END_FRAC * W)

    return y0, y1, x0, x1


def normalize_slice_and_profiles(raw_slice, y0, y1, x0, x1, scale=10000.0):
    """
    raw_slice: 2D-Array mit Roh-Counts (H, W).
    Gibt zurueck:
      horiz_norm, vert_norm, horiz_err_norm, vert_err_norm, norm_factor
    """
    # Normierung wie in normalize_like_validation pro Slice
    total_counts = np.sum(raw_slice) + 1e-12
    norm_factor = scale / total_counts

    slice_norm = raw_slice * norm_factor

    # ROI (normiert) -> gemittelte Profile
    roi_norm = slice_norm[y0:y1, x0:x1]
    horiz_norm = roi_norm.mean(axis=0)
    vert_norm  = roi_norm.mean(axis=1)

    # ROI (roh) -> Summen pro Pixel-Spalte/-Zeile fuer Poisson-Fehler
    roi_raw = raw_slice[y0:y1, x0:x1]
    horiz_raw = roi_raw.sum(axis=0)  # Summe ueber y -> Counts pro x
    vert_raw  = roi_raw.sum(axis=1)  # Summe ueber x -> Counts pro y

    horiz_err_norm = np.sqrt(horiz_raw) * norm_factor
    vert_err_norm  = np.sqrt(vert_raw)  * norm_factor

    return horiz_norm, vert_norm, horiz_err_norm, vert_err_norm, norm_factor


def gaussian(x, A, x0, sigma, c):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c


def fit_gaussian(x, y):
    """
    Einfache Gauss-Fit-Hilfsfunktion.
    Gibt (popt, pcov) zurueck, oder (None, None) falls Fit scheitert.
    """
    A0 = float(y.max() - y.min())
    x0_0 = float(x[np.argmax(y)])
    sigma0 = max(1.0, 0.1 * (x.max() - x.min()))
    c0 = float(y.min())
    p0 = [A0, x0_0, sigma0, c0]

    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=10000)
    except Exception as e:
        print("Gauss-Fit fehlgeschlagen:", e)
        popt, pcov = None, None
    return popt, pcov

# =====================================================
# Plot-Funktion
# =====================================================
def plot_gauss_panels(x_h, profiles_h, x_v, profiles_v, title_suffix, out_dir):
    """
    profiles_h / profiles_v: dict mit Keys 'LC','Pred','HC'
    und Werten (y, yerr).
    Erzeugt eine Figur mit 3x2 Panels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), dpi=150, sharex=False)

    labels = ["LC (Input)", "Prediction", "HC (Ground Truth)"]
    keys   = ["LC", "Pred", "HC"]

    for row, (key, label) in enumerate(zip(keys, labels)):
        # Horizontal
        ax_h = axes[row, 0]
        y_h, yerr_h = profiles_h[key]
        ax_h.errorbar(x_h, y_h, yerr=yerr_h, fmt='o', markersize=3, linewidth=1)

        popt_h, _ = fit_gaussian(x_h, y_h)
        if popt_h is not None:
            x_dense = np.linspace(x_h.min(), x_h.max(), 500)
            ax_h.plot(x_dense, gaussian(x_dense, *popt_h), '-', linewidth=1.5)
        ax_h.set_ylabel("Intensität (a.u.)")
        if row == 2:
            ax_h.set_xlabel("Pixel (x) in ROI")
        ax_h.set_title(f"Horizontal – {label}")
        if H_X_LIM is not None:
            ax_h.set_xlim(*H_X_LIM)
        if H_Y_LIM is not None:
            ax_h.set_ylim(*H_Y_LIM)

        # Vertikal
        ax_v = axes[row, 1]
        y_v, yerr_v = profiles_v[key]
        ax_v.errorbar(x_v, y_v, yerr=yerr_v, fmt='o', markersize=3, linewidth=1)

        popt_v, _ = fit_gaussian(x_v, y_v)
        if popt_v is not None:
            x_dense = np.linspace(x_v.min(), x_v.max(), 500)
            ax_v.plot(x_dense, gaussian(x_dense, *popt_v), '-', linewidth=1.5)
        if row == 2:
            ax_v.set_xlabel("Pixel (y) in ROI")
        ax_v.set_title(f"Vertikal – {label}")
        if V_X_LIM is not None:
            ax_v.set_xlim(*V_X_LIM)
        if V_Y_LIM is not None:
            ax_v.set_ylim(*V_Y_LIM)

    fig.suptitle(f"Gauss-Fits in ROI – {title_suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = out_dir / f"{title_suffix}_gauss_fits.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out_path}")

# =====================================================
# Hauptfunktion
# =====================================================
def main():
    print(f"Modell-Datei:  {MODEL_PATH}")
    print(f"Test-HDF5:     {H5_TEST_PATH}")
    print(f"Serie:         {SERIES_IDX}, Volumen: {TARGET_VOL}")
    print(f"Output-Dir:    {OUT_DIR}")

    print("Lade Modell...")
    model = models.load_model(MODEL_PATH, compile=False)

    print("Lade Testdaten (roh)...")
    X_test, y_test = load_test_split(H5_TEST_PATH)

    print("Erzeuge Volumen per Sliding Window...")
    X_vols_raw, Y_vols_raw, n_series, n_vols_per_series = make_sliding_windows(
        X_test, y_test, SERIES_LEN, DEPTH
    )
    assert n_vols_per_series == N_VOLS_PER_SERIES

    # globaler Volumen-Index
    global_vol_idx = SERIES_IDX * n_vols_per_series + TARGET_VOL
    assert 0 <= global_vol_idx < X_vols_raw.shape[0], "TARGET_VOL ausserhalb des Bereichs"

    # Nur dieses eine Volumen verwenden (roh)
    X_target_raw = X_vols_raw[global_vol_idx:global_vol_idx+1]  # (1, D, H, W, 1)
    Y_target_raw = Y_vols_raw[global_vol_idx:global_vol_idx+1]

    # Normierte Volumen fuer das Netz
    print("Normalisiere Volume fuer das Netz...")
    X_norm = normalize_like_validation(X_target_raw, scale=10000.0)
    # (Y_norm waere nur fuer Training wichtig; fuer die Fits verwenden wir Rohdaten)
    # Y_norm = normalize_like_validation(Y_target_raw, scale=10000.0)

    print("Berechne Prediction nur fuer dieses Volumen...")
    Y_pred = model.predict(X_norm, batch_size=1, verbose=0)

    # Zentralslice (roh) fuer LC/HC
    center_idx = DEPTH // 2
    raw_lc_slice = X_target_raw[0, center_idx, :, :, 0]
    raw_hc_slice = Y_target_raw[0, center_idx, :, :, 0]

    H, W = raw_lc_slice.shape
    y0, y1, x0, x1 = get_roi_indices(H, W)
    print(f"ROI: y=[{y0}:{y1}), x=[{x0}:{x1}) (H={H}, W={W})")

    # Normierte Profile + Fehler aus Rohdaten (LC/HC)
    horiz_lc, vert_lc, err_h_lc, err_v_lc, nf_lc = normalize_slice_and_profiles(
        raw_lc_slice, y0, y1, x0, x1, scale=10000.0
    )
    horiz_hc, vert_hc, err_h_hc, err_v_hc, nf_hc = normalize_slice_and_profiles(
        raw_hc_slice, y0, y1, x0, x1, scale=10000.0
    )

    # Prediction: Slice in derselben Normierung wie das Netz ausgibt
    pred_slice_norm = Y_pred[0, 0, :, :, 0]
    roi_pred = pred_slice_norm[y0:y1, x0:x1]
    horiz_pr = roi_pred.mean(axis=0)
    vert_pr  = roi_pred.mean(axis=1)
    # Prediction hat keine Poisson-Fehler -> hier 0
    err_h_pr = np.zeros_like(horiz_pr)
    err_v_pr = np.zeros_like(vert_pr)

    # Subsampling
    x_h_full = np.arange(x0, x1)  # Pixel-Indizes innerhalb des ganzen Bildes (optional)
    x_v_full = np.arange(y0, y1)

    # Wenn du lieber 0..(Breite-1) willst, mache:
    x_h_roi = np.arange(x1 - x0)
    x_v_roi = np.arange(y1 - y0)

    # wir nehmen hier 0..Breite-1:
    x_h = x_h_roi[::STEP_PROFILE]
    x_v = x_v_roi[::STEP_PROFILE]

    horiz_lc_s = horiz_lc[::STEP_PROFILE]
    horiz_pr_s = horiz_pr[::STEP_PROFILE]
    horiz_hc_s = horiz_hc[::STEP_PROFILE]

    vert_lc_s = vert_lc[::STEP_PROFILE]
    vert_pr_s = vert_pr[::STEP_PROFILE]
    vert_hc_s = vert_hc[::STEP_PROFILE]

    err_h_lc_s = err_h_lc[::STEP_PROFILE]
    err_h_pr_s = err_h_pr[::STEP_PROFILE]
    err_h_hc_s = err_h_hc[::STEP_PROFILE]

    err_v_lc_s = err_v_lc[::STEP_PROFILE]
    err_v_pr_s = err_v_pr[::STEP_PROFILE]
    err_v_hc_s = err_v_hc[::STEP_PROFILE]

    profiles_h = {
        "LC":   (horiz_lc_s, err_h_lc_s),
        "Pred": (horiz_pr_s, err_h_pr_s),
        "HC":   (horiz_hc_s, err_h_hc_s),
    }
    profiles_v = {
        "LC":   (vert_lc_s, err_v_lc_s),
        "Pred": (vert_pr_s, err_v_pr_s),
        "HC":   (vert_hc_s, err_v_hc_s),
    }

    title_suffix = f"series{SERIES_IDX}_vol{TARGET_VOL:02d}"
    plot_gauss_panels(x_h, profiles_h, x_v, profiles_v, title_suffix, OUT_DIR)

    print("Fertig mit Gauss-Fits.")

if __name__ == "__main__":
    main()
