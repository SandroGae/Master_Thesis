#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits" # Originaler Pfad
NPZ_FILE = "Pred_unet_25d_SSIM_middle_improved_V2_D5_S12_FullSeries.npz"

# Settings
SLICE_INDEX = 19
ROI_X_START, ROI_X_END = 0, 240   
ROI_Y_START, ROI_Y_END = 102, 117  
BACKGROUND_GAP        = 5   
BACKGROUND_BOX_HEIGHT = 10  
FIT_WINDOW_X  = (20, 120)
FIT_COLORS     = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES         = ["Low Count", "Prediction", "Ground Truth"]

# Berechne Koordinaten Background boxes
red_region1_bottom = max(0, ROI_Y_START - BACKGROUND_GAP)
red_region1_top    = max(0, red_region1_bottom - BACKGROUND_BOX_HEIGHT)
red_region2_top    = ROI_Y_END + BACKGROUND_GAP
red_region2_bottom = red_region2_top + BACKGROUND_BOX_HEIGHT


def vis_norm(image):
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    clipped_image = np.clip(image, vmin, vmax)
    if vmax - vmin == 0: return clipped_image
    clipped_value = (clipped_image - vmin) / (vmax - vmin)
    return clipped_value

def get_background_std(image, ROI_x_start, ROI_x_end):
    background_pixels = []
    box_top = image[red_region1_top:red_region1_bottom, ROI_x_start:ROI_x_end]
    background_pixels.append(box_top)
    box_bottom = image[red_region2_top:red_region2_bottom, ROI_x_start:ROI_x_end]
    background_pixels.append(box_bottom)

    combined_background = np.concatenate(background_pixels, axis=0)
    std_background = np.std(combined_background, axis=0)
    return std_background


def calculate_sbr_profiles(image, force_noise_std=None):
    """
    Berechnet SRBR = (Signal - Background) / Background
    """
    # 1. Summe Signal
    signal_slice = image[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
    n_signal_rows = signal_slice.shape[0]
    profile_signal = np.sum(signal_slice, axis=0)
    
    # 2. Background Daten holen
    background_list = []
    background_list.append(image[red_region1_top:red_region1_bottom, ROI_X_START:ROI_X_END])
    background_list.append(image[red_region2_top:red_region2_bottom, ROI_X_START:ROI_X_END])
    background_slice = np.concatenate(background_list, axis=0)
    n_background_rows = background_slice.shape[0]
    
    # 3. Background aggregieren (skalieren auf Signalgröße)
    profile_background_raw = np.sum(background_slice, axis=0)
    scale = n_signal_rows / n_background_rows if n_background_rows > 0 else 0
    profile_background = profile_background_raw * scale

    # 4. Netto Signal (A)
    profile_net = profile_signal - profile_background

    # 5. SRBR Berechnung (A / C)
    denom = profile_background.copy()
    denom[denom == 0] = 1e-9 # Schutz vor Div/0
    profile_sbr = profile_net / denom

    # 6. Fehlerberechnung
    if force_noise_std is not None:
        pixel_noise_std = force_noise_std
    else:
        pixel_noise_std = np.std(background_slice, axis=0) 
        
    err_signal_sum = pixel_noise_std * np.sqrt(n_signal_rows)
    err_bg_sum     = pixel_noise_std * np.sqrt(n_background_rows) * scale
    
    # Fehler des Netto-Signals (absolut)
    err_net = np.sqrt(err_signal_sum**2 + err_bg_sum**2)
    
    # Fehler des SRBR (relativ fortgepflanzt)
    safe_net = profile_net.copy()
    safe_net[safe_net == 0] = 1.0 
    
    rel_err_net = err_net / np.abs(safe_net)
    rel_err_bg  = err_bg_sum / np.abs(denom)
    
    total_rel_err = np.sqrt(rel_err_net**2 + rel_err_bg**2)
    profile_sbr_error = np.abs(profile_sbr) * total_rel_err
    
    x = np.arange(ROI_X_START, ROI_X_END)
    
    box_coords = ((red_region1_top, red_region1_bottom), (red_region2_top, red_region2_bottom))
    
    return x, profile_signal, profile_background, profile_sbr, profile_sbr_error, box_coords


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit = x[mask]
    y_fit = y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    valid = np.isfinite(y_fit)
    if np.sum(valid) < 4: return None, None
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    if sigma_fit is not None: sigma_fit = sigma_fit[valid]

    A  = np.max(y_fit) - np.min(y_fit) 
    x0 = x_fit[np.argmax(y_fit)]
    s  = (np.max(x_fit) - np.min(x_fit)) * 0.1
    p0 = [A, x0, s]
    
    try:
        parameters, _ = curve_fit(gaussian, x_fit, y_fit, p0, sigma=sigma_fit, absolute_sigma=True, maxfev=5000)
        return gaussian(x, *parameters), parameters
    except:
        return None, None


def main():
    file_path = IN_DIR / NPZ_FILE 
    data = np.load(file_path)
    
    raw_lc = data['lc']
    if raw_lc.ndim == 3:
        images = [data['lc'][SLICE_INDEX], data['pred'][SLICE_INDEX], data['gt'][SLICE_INDEX]] 
    else:
        images = [data['lc'], data['pred'], data['gt']] 
    
    gt_noise_std = get_background_std(images[2], ROI_X_START, ROI_X_END) 
    
    results = []
    for index, image in enumerate(images):
        noise_to_use = gt_noise_std if index == 1 else None
        
        # SBR Berechnung
        x_axis, signal, background, sbr, error, boxes = calculate_sbr_profiles(
            image, force_noise_std=noise_to_use
        )
        
        # Fit auf SBR
        fit_y, params = perform_gaussian_fit(x_axis, sbr, y_err=error, fit_window=FIT_WINDOW_X)
        
        results.append({
            'x_axis':x_axis, 'signal':signal, 'background':background, 
            'sbr':sbr, 'error': error,
            'fit':fit_y, 'par':params, 'boxes':boxes
        })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    for i in range(3):
        # --- 1. Bild ---
        ax = axes[0, i]
        ax.imshow(vis_norm(images[i]), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.axis('off')
        
        (r1_top, r1_bottom), (r2_top, r2_bottom) = results[i]['boxes']
        roi_w = ROI_X_END - ROI_X_START
        roi_h = ROI_Y_END - ROI_Y_START
        ax.add_patch(patches.Rectangle((ROI_X_START, ROI_Y_START), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        fit_w = FIT_WINDOW_X[1] - FIT_WINDOW_X[0]
        ax.add_patch(patches.Rectangle((FIT_WINDOW_X[0], ROI_Y_START), fit_w, roi_h, lw=0, fc='green', alpha=0.2))
        
        # --- FIX: Berechnung der Höhen ---
        height_1 = r1_bottom - r1_top
        height_2 = r2_bottom - r2_top
        
        ax.add_patch(patches.Rectangle((ROI_X_START, r1_top), roi_w, height_1, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((ROI_X_START, r2_top), roi_w, height_2, lw=1, ec='red', fc='red', alpha=0.2))

        # --- 2. Totale Intensitäten (Bleibt absolut) ---
        ax2 = axes[1, i]
        ax2.plot(results[i]['x_axis'], results[i]['signal'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x_axis'], results[i]['background'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW_X[0], FIT_WINDOW_X[1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        if i==0: ax2.set_ylabel("Total Counts (Abs)")
        if i==1: ax2.legend(loc='upper right', fontsize=8)

        # --- 3. SRBR ---
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x_axis'], results[i]['sbr'], 
                     yerr=results[i]['error'], 
                     fmt='.', markersize=5, elinewidth=2, capsize=1, color='black', alpha=0.6, label='SRBR Data')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if i == 1: 
            if results[0]['fit'] is not None: ax3.plot(results[0]['x_axis'], results[0]['fit'], color=FIT_COLORS[0], ls=':', lw=1.5, label='LC Fit')
            if results[2]['fit'] is not None: ax3.plot(results[2]['x_axis'], results[2]['fit'], color=FIT_COLORS[2], ls=':', lw=1.5, label='GT Fit') 
            if results[1]['fit'] is not None:
                l = f"Pred (Max SRBR={np.max(results[1]['fit']):.2f})"
                ax3.plot(results[1]['x_axis'], results[1]['fit'], color=FIT_COLORS[1], ls='--', lw=2.5, label=l)
            ax3.set_title("Comparison: SRBR Fits")
        else: 
            if results[i]['fit'] is not None:
                l = f"Gauss (Max SRBR={np.max(results[i]['fit']):.2f})"
                ax3.plot(results[i]['x_axis'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
            ax3.set_title("SRBR & Fit")
            
        ax3.set_xlabel("Pixel X")
        if i==0: ax3.set_ylabel("SRBR (A/C)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        
        ax3.set_xlim(FIT_WINDOW_X)
        mask = (results[i]['x_axis'] >= FIT_WINDOW_X[0]) & (results[i]['x_axis'] <= FIT_WINDOW_X[1])
        if np.any(mask):
            y_vis = results[i]['sbr'][mask]
            y_vis = y_vis[np.isfinite(y_vis)]
            if len(y_vis) > 0:
                ymin, ymax = np.min(y_vis), np.max(y_vis)
                rng = ymax - ymin if (ymax-ymin) > 0.1 else 1.0
                ax3.set_ylim(ymin - 0.2 * rng, ymax + 0.2 * rng)

    plt.tight_layout()
    out_name = f"Ana_{Path(NPZ_FILE).stem}_Slice{SLICE_INDEX}_SRBR_l.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot fertig: {out_name}")

if __name__ == "__main__":
    main()