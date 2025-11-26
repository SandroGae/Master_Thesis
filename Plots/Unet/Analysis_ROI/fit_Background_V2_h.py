#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits"
NPZ_FILE = "Pred_unet_25d_SSIM_middle_improved_V2_D5_S12_FullSeries.npz" # Datei mit ganzer Serie

# Settings
SLICE_INDEX = 19 # Wähle Bild für Visualisierung oben
ROI_X_START, ROI_X_END = 65, 86   
ROI_Y_START, ROI_Y_END = 104, 115  
BACKGROUND_GAP        = 5   
BACKGROUND_BOX_HEIGHT = 10  
FIT_WINDOW_FRAMES = (2, 38) # Bereich für Gauss Fit (Frames/Indizes)
FIT_COLORS     = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES         = ["Low Count", "Prediction", "Ground Truth"]

# Berechne Koordinaten Background boxes
bg_top_y_bottom = max(0, ROI_Y_START - BACKGROUND_GAP)
bg_top_y_top    = max(0, bg_top_y_bottom - BACKGROUND_BOX_HEIGHT)
bg_bot_y_top    = ROI_Y_END + BACKGROUND_GAP
bg_bot_y_bottom = bg_bot_y_top + BACKGROUND_BOX_HEIGHT


def vis_norm(image):
    """
    Visualisiert die Bilder durch clipping und Normalisierung
    """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin == 0: return image
    clipped_image = np.clip(image, vmin, vmax)
    clipped_value = (clipped_image - vmin) / (vmax - vmin)
    return clipped_value

def get_gt_noise_std_series(volume):
    """
    Hilfsfunktion: Extrahiert die Std-Abweichung des Hintergrunds für alle Frames im GT Volume.
    """
    std_list = []
    n_frames = volume.shape[0]
    for i in range(n_frames):
        img = volume[i]
        bg_pixels = []
        # Top Box
        if bg_top_y_bottom > bg_top_y_top:
            bg_pixels.append(img[bg_top_y_top:bg_top_y_bottom, ROI_X_START:ROI_X_END])
        # Bottom Box
        if bg_bot_y_bottom > bg_bot_y_top and bg_bot_y_bottom <= img.shape[0]:
            bg_pixels.append(img[bg_bot_y_top:bg_bot_y_bottom, ROI_X_START:ROI_X_END])
        
        if bg_pixels:
            combined = np.concatenate(bg_pixels)
            std_list.append(np.std(combined))
        else:
            std_list.append(1e-9)
    return np.array(std_list)


def calculate_sbr_z_profiles(volume, force_noise_std_array=None):
    """
    NEU: Berechnet SRBR = (Signal - Background) / Background über die Z-Achse (Frames)
    """
    n_frames = volume.shape[0]
    
    profile_signal = []
    profile_background = []
    profile_sbr = []
    profile_sbr_error = []
    
    # Anzahl Pixel im Signal ROI (konstant)
    n_signal_pixels = (ROI_X_END - ROI_X_START) * (ROI_Y_END - ROI_Y_START)

    for i in range(n_frames):
        image = volume[i]
        
        # 1. Summe Signal
        signal_slice = image[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        sum_signal = np.sum(signal_slice)
        
        # 2. Background Daten holen
        bg_pixels_list = []
        
        # Top Box
        t_y1, t_y2 = bg_top_y_top, bg_top_y_bottom
        if t_y2 > t_y1: bg_pixels_list.append(image[t_y1:t_y2, ROI_X_START:ROI_X_END])
        
        # Bottom Box - Check Boundaries
        h = image.shape[0]
        b_y1 = min(h, bg_bot_y_top)
        b_y2 = min(h, bg_bot_y_bottom)
        if b_y2 > b_y1: bg_pixels_list.append(image[b_y1:b_y2, ROI_X_START:ROI_X_END])
        
        if bg_pixels_list:
            bg_concat = np.concatenate(bg_pixels_list)
            mean_bg = np.mean(bg_concat)
            n_background_pixels = bg_concat.size
            
            sum_bg_equivalent = mean_bg * n_signal_pixels   # Skalierung auf Signal-Grösse
            scale = n_signal_pixels / n_background_pixels
        
            net_signal = sum_signal - sum_bg_equivalent # Netto Signal
            
            denom = sum_bg_equivalent if sum_bg_equivalent > 1e-9 else 1e-9 # SRBR
            sbr = net_signal / denom

            # Rauschen und Fehler
            if force_noise_std_array is not None:
                pixel_noise_std = force_noise_std_array[i]
            else:
                pixel_noise_std = np.std(bg_concat)
            
            err_signal = pixel_noise_std * np.sqrt(n_signal_pixels)
            err_bg = pixel_noise_std * np.sqrt(n_background_pixels) * scale
            
            err_net = np.sqrt(err_signal**2 + err_bg**2) # Fehler Netto Signal
            safe_net = net_signal if abs(net_signal) > 1e-9 else 1.0 # Relativer Fehler für SRBR
            
            rel_err_net = err_net / abs(safe_net)
            rel_err_bg  = err_bg / abs(denom)
            
            total_rel = np.sqrt(rel_err_net**2 + rel_err_bg**2)
            sbr_error_val = abs(sbr) * total_rel

        else:
            sum_bg_equivalent = 0
            sum_signal = 0
            sbr = 0
            sbr_error_val = 0

        profile_signal.append(sum_signal)
        profile_background.append(sum_bg_equivalent)
        profile_sbr.append(sbr)
        profile_sbr_error.append(sbr_error_val)

    x = np.arange(n_frames) # Convert to arrays
    box_coords = ((bg_top_y_top, bg_top_y_bottom), (bg_bot_y_top, bg_bot_y_bottom)) # Rückgabe der Box-Koordinaten
    
    return x, np.array(profile_signal), np.array(profile_background), np.array(profile_sbr), np.array(profile_sbr_error), box_coords


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit = x[mask]
    y_fit = y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    # Valid Check für SRBR
    valid = np.isfinite(y_fit)
    if np.sum(valid) < 4: 
        return None, None, None
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    if sigma_fit is not None: sigma_fit = sigma_fit[valid]

    A  = np.max(y_fit)                                         
    x0 = x_fit[np.argmax(y_fit)]                           
    s  = (np.max(x_fit) - np.min(x_fit)) * 0.2             
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
    volumes = [data['lc'], data['pred'], data['gt']] # Volumes laden
    gt_noise_series = get_gt_noise_std_series(volumes[2]) 
    
    results = []
    viz_boxes = None 

    for index, vol in enumerate(volumes):
        # Prediction (Index 1) nutzt GT Rauschen
        noise_to_use = gt_noise_series if index == 1 else None
        
        # Aufruf der NEUEN SBR Funktion (Z-Achse)
        x_axis, signal, background, sbr, error, boxes = calculate_sbr_z_profiles(vol, force_noise_std_array=noise_to_use)
        viz_boxes = boxes 
        
        # Fit auf SRBR
        fit_y, params, perr = perform_gaussian_fit(x_axis, sbr, y_err=error, fit_window=FIT_WINDOW_FRAMES)
        
        results.append({
            'x_axis':x_axis, 'signal':signal, 'background':background, 
            'sbr':sbr, 'error': error,
            'fit':fit_y, 'par':params, 'perr':perr}) # <-- perr speichern

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    for i in range(3):
        # Bild
        ax = axes[0, i]
        img_show = volumes[i][SLICE_INDEX]
        ax.imshow(vis_norm(img_show), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Koordinaten
        (r1_top, r1_bottom), (r2_top, r2_bottom) = viz_boxes
        roi_w = ROI_X_END - ROI_X_START
        roi_h = ROI_Y_END - ROI_Y_START
        
        # Grüner Hintergrund (Integration Area)
        ax.add_patch(patches.Rectangle((ROI_X_START, ROI_Y_START), roi_w, roi_h, lw=0, fc='green', alpha=0.3))
        # Blaue ROI Box
        ax.add_patch(patches.Rectangle((ROI_X_START, ROI_Y_START), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        
        # Rote Background Boxen
        h1 = r1_bottom - r1_top
        h2 = r2_bottom - r2_top
        if h1 > 0: ax.add_patch(patches.Rectangle((ROI_X_START, r1_top), roi_w, h1, lw=1, ec='red', fc='red', alpha=0.2))
        if h2 > 0: ax.add_patch(patches.Rectangle((ROI_X_START, r2_top), roi_w, h2, lw=1, ec='red', fc='red', alpha=0.2))

        # Totale Intensitäten (Bleibt absolut)
        ax2 = axes[1, i]
        ax2.plot(results[i]['x_axis'], results[i]['signal'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x_axis'], results[i]['background'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW_FRAMES[0], FIT_WINDOW_FRAMES[1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        if i==0: 
            ax2.set_ylabel("Integrated Counts")
        if i==1: 
            ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(42.5, 57.5)

        # SRBR
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x_axis'], results[i]['sbr'], 
                     yerr=results[i]['error'], 
                     fmt='.', markersize=8, elinewidth=2, capsize=2, color='black', alpha=0.6, label='SRBR Data')
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
            
        ax3.set_xlabel("Image Index (Z-Axis)")
        if i==0: 
            ax3.set_ylabel("SRBR: (Signal - Background) / Background")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        
        # Scaling
        margin = 2
        ax3.set_xlim(max(0, FIT_WINDOW_FRAMES[0]-margin), results[i]['x_axis'][-1])
        ax3.set_ylim(-0.1, 0.4)

    plt.tight_layout()
    out_name = f"Ana_{Path(NPZ_FILE).stem}_Slice{SLICE_INDEX}_SRBR_h.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot fertig: {out_name}")

if __name__ == "__main__":
    main()