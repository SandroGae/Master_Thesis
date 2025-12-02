#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits"
NPZ_FILE = "Pred_FILE_25d_middle_improved_V2_interpolated_D5_S12_FullSeries.npz" # Datei mit ganzer Serie

# Settings
SLICE_INDEX = 19 # Wähle Bild
IMAGE_WIDTH = 240 
ROI_X_START, ROI_X_END = 60, 81   
ROI_Y_START, ROI_Y_END = 0, 192  
BACKGROUND_GAP        = 80
BACKGROUND_BOX_WIDTH  = 10  
FIT_WINDOW_Y  = (90, 130) # Gauss Fit Bereich
FIT_COLORS     = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES         = ["Low Count", "Prediction", "Ground Truth"]

# Berechne Koordinaten Background boxes
# Wir nehmen rechts die doppelte Breite, um Statistik anzugleichen
red_region_left  = min(IMAGE_WIDTH, ROI_X_END + BACKGROUND_GAP)
red_region_right = min(IMAGE_WIDTH, red_region_left + 2 * BACKGROUND_BOX_WIDTH)


def vis_norm(image):
    """
    Visualisiert die Bilder durch clipping und Normalisierung
    """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    clipped_image = np.clip(image, vmin, vmax)
    if vmax - vmin == 0: return clipped_image
    clipped_value = (clipped_image - vmin) / (vmax - vmin)
    return clipped_value

def get_background_std(image, ROI_y_start, ROI_y_end):
    """
    Extrahiert die Standardabweichung des Hintergrundrauschens in den roten Boxen
    """
    background_pixels = []
    # Box rechts
    box_right = image[ROI_y_start:ROI_y_end, red_region_left:red_region_right]
    background_pixels.append(box_right)

    combined_background = np.concatenate(background_pixels, axis=0)
    std_background = np.std(combined_background)

    return std_background


def calculate_sbr_k_profiles(image, force_noise_std=None):
    """
    NEU: Berechnet SRBR = (Signal - Background) / Background entlang der Y-Achse (k)
    """
    # 1. Summe Signal (entlang Zeilen -> axis=1)
    signal_slice = image[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
    n_signal_cols = signal_slice.shape[1]
    profile_signal = np.sum(signal_slice, axis=1)
    
    # 2. Background Daten holen
    background_slice = image[ROI_Y_START:ROI_Y_END, red_region_left:red_region_right]
    n_background_cols = background_slice.shape[1]
    
    # 3. Background arrays aggregieren
    profile_background_raw = np.sum(background_slice, axis=1)
    scale = n_signal_cols / n_background_cols if n_background_cols > 0 else 0
    profile_background = profile_background_raw * scale

    # 4. Netto Signal (A)
    profile_net = profile_signal - profile_background

    # 5. SRBR Berechnung (A / C)
    denom = profile_background.copy()
    denom[denom == 0] = 1e-9 # Schutz Division durch 0
    profile_sbr = profile_net / denom

    # 6. Fehlerberechnung
    if force_noise_std is not None:
        pixel_noise_std = force_noise_std
    else:
        pixel_noise_std = np.std(background_slice) 
        
    err_signal_sum = pixel_noise_std * np.sqrt(n_signal_cols)
    err_bg_sum     = pixel_noise_std * np.sqrt(n_background_cols) * scale
    
    # Skalarer Fehler für diesen Slice (wird auf Array erweitert)
    scalar_err_net = np.sqrt(err_signal_sum**2 + err_bg_sum**2)
    profile_err_net = np.full_like(profile_signal, scalar_err_net)
    
    # Relativer Fehler für SRBR Fortpflanzung
    safe_net = profile_net.copy()
    safe_net[safe_net == 0] = 1.0
    
    rel_err_net = profile_err_net / np.abs(safe_net)
    # Background Fehler ist hier eigentlich auch ein Array (konstant skaliert), 
    # aber wir nutzen den Fehler der Summe
    scalar_err_bg = err_bg_sum 
    rel_err_bg = scalar_err_bg / np.abs(denom)
    
    total_rel_err = np.sqrt(rel_err_net**2 + rel_err_bg**2)
    profile_sbr_error = np.abs(profile_sbr) * total_rel_err
    
    x = np.arange(ROI_Y_START, ROI_Y_END)
    box_coords = (red_region_left, red_region_right)
    
    return x, profile_signal, profile_background, profile_sbr, profile_sbr_error, box_coords


def gaussian(x, amplitude, mu, sigma):
    gauss = amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return gauss

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit = x[mask]
    y_fit = y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    # Valid Check für SRBR (NaNs/Infs filtern)
    valid = np.isfinite(y_fit)
    if np.sum(valid) < 4: 
        return None, None, None
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    if sigma_fit is not None: sigma_fit = sigma_fit[valid]

    A  = np.max(y_fit) - np.min(y_fit)                 
    x0 = x_fit[np.argmax(y_fit)]                       
    s  = (np.max(x_fit) - np.min(x_fit)) * 0.1         
    p0 = [A, x0, s]
    
    try:
        parameters, pcov = curve_fit(gaussian, x_fit, y_fit, p0, sigma=sigma_fit, absolute_sigma=True, maxfev=5000)
        perr = np.sqrt(np.diag(pcov)) # Fehler berechnen
        return gaussian(x, *parameters), parameters, perr 
    except:
        return None, None, None



def main():
    file_path = IN_DIR / NPZ_FILE 
    data = np.load(file_path)
    
    # Slice auswählen
    raw_lc = data['lc']
    if raw_lc.ndim == 3:
        images = [data['lc'][SLICE_INDEX], data['pred'][SLICE_INDEX], data['gt'][SLICE_INDEX]] 
    else:
        images = [data['lc'], data['pred'], data['gt']] 
    
    # Rauschen von Ground Truth (Index 2) holen
    gt_noise_std = get_background_std(images[2], ROI_Y_START, ROI_Y_END) 
    
    results = []
    for index, image in enumerate(images):
        # Prediction (Index 1) nutzt GT Rauschen
        noise_to_use = gt_noise_std if index == 1 else None
        x_axis, signal, background, sbr, error, boxes = calculate_sbr_k_profiles(image, force_noise_std=noise_to_use)
        
        # Fit auf SRBR
        fit_y, params, perr = perform_gaussian_fit(x_axis, sbr, y_err=error, fit_window=FIT_WINDOW_Y)
        
        results.append({
            'x_axis':x_axis, 'signal':signal, 'background':background, 
            'sbr':sbr, 'error': error,
            'fit':fit_y, 'par':params, 'perr':perr, 'boxes':boxes # <-- perr speichern
        })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    for i in range(3):
        # Bild
        ax = axes[0, i]
        ax.imshow(vis_norm(images[i]), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Background Box Koordinaten
        r_left, r_right = results[i]['boxes']
        bg_width = r_right - r_left

        # Blaue ROI Box
        roi_w = ROI_X_END - ROI_X_START
        roi_h = ROI_Y_END - ROI_Y_START
        ax.add_patch(patches.Rectangle((ROI_X_START, ROI_Y_START), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        
        # Grüner Fit Bereich (Horizontaler Streifen)
        fit_h = FIT_WINDOW_Y[1] - FIT_WINDOW_Y[0]
        ax.add_patch(patches.Rectangle((ROI_X_START, FIT_WINDOW_Y[0]), roi_w, fit_h, lw=0, fc='green', alpha=0.2))
        
        # Rote Background Boxen (Nur Rechts)
        if bg_width > 0:
            ax.add_patch(patches.Rectangle((r_left, ROI_Y_START), bg_width, roi_h, lw=1, ec='red', fc='red', alpha=0.2))

        # Intensitäten (Bleibt absolut)
        ax2 = axes[1, i]
        ax2.plot(results[i]['x_axis'], results[i]['signal'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x_axis'], results[i]['background'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW_Y[0], FIT_WINDOW_Y[1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        if i==0: 
            ax2.set_ylabel("Total Counts")
        if i==1: 
            ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(2.5, 7)

        # SRBR
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x_axis'], results[i]['sbr'], 
                     yerr=results[i]['error'], 
                     fmt='.', markersize=5, elinewidth=2, capsize=1, color='black', alpha=0.6, label='SRBR Data')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if i == 1: 
            # Prediction fit grün
            if results[1]['fit'] is not None:
                sigma_val = results[1]['par'][2] # Parameter index 2 ist sigma
                sigma_err = results[1]['perr'][2] # Fehler holen
                l = f"Pred (Max={np.max(results[1]['fit']):.2f}, $\sigma$={sigma_val:.2f}$\pm${sigma_err:.2f})"
                ax3.plot(results[1]['x_axis'], results[1]['fit'], color=FIT_COLORS[1], ls='--', lw=2.5, label=l)
        else: 
            # LC (i=0) und GT (i=2) Einzelplots
            if results[i]['fit'] is not None:
                sigma_val = results[i]['par'][2]
                sigma_err = results[i]['perr'][2]
                l = f"Gauss (Max={np.max(results[i]['fit']):.2f}, $\sigma$={sigma_val:.2f}$\pm${sigma_err:.2f})"
                ax3.plot(results[i]['x_axis'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
            
        ax3.set_xlabel("Pixel Y")
        if i==0: 
            ax3.set_ylabel("SRBR: (Signal - Background) / Background")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        
        ax3.set_xlim(FIT_WINDOW_Y)
        ax3.set_ylim(-0.1, 0.4)

    plt.tight_layout()
    out_name = f"Ana_{Path(NPZ_FILE).stem}_Slice{SLICE_INDEX}_SRBR_k.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot fertig: {out_name}")

if __name__ == "__main__":
    main()