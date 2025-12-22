#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits" # Originaler Pfad
NPZ_FILE = "Pred_FILE_25d_middle_improved_V2_interpolated_D5_S12_FullSeries.npz"

"""
# Settings series 12
SLICE_INDEX = 19
ROI_X_START, ROI_X_END = 0, 240   
ROI_Y_START, ROI_Y_END = 102, 117  
BACKGROUND_GAP        = 5   
BACKGROUND_BOX_HEIGHT = 10  
FIT_WINDOW_X  = (20, 120)
FIT_COLORS     = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES         = ["Low Count", "Prediction", "Ground Truth"]
"""

# Settings series 29
NPZ_FILE = "Pred_FILE_25d_middle_improved_V2_interpolated_D5_S29_FullSeries.npz" # Datei mit ganzer Serie
SLICE_INDEX = 24 # Wähle Bild für Visualisierung oben
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
    # Summe Signal
    signal_slice = image[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
    n_signal_rows = signal_slice.shape[0]
    profile_signal = np.sum(signal_slice, axis=0)
    
    # Background Daten holen
    background_list = []
    background_list.append(image[red_region1_top:red_region1_bottom, ROI_X_START:ROI_X_END])
    background_list.append(image[red_region2_top:red_region2_bottom, ROI_X_START:ROI_X_END])
    background_slice = np.concatenate(background_list, axis=0)
    n_background_rows = background_slice.shape[0]
    
    # Background aggregieren (skalieren auf Signalgröße)
    profile_background_raw = np.sum(background_slice, axis=0)
    scale = n_signal_rows / n_background_rows if n_background_rows > 0 else 0
    profile_background = profile_background_raw * scale

    # Netto Signal
    profile_net = profile_signal - profile_background

    # SRBR Berechnung
    denom = profile_background.copy()
    denom[denom == 0] = 1e-9 # Schutz vor Div/0
    profile_sbr = profile_net / denom

    # Fehlerberechnung
    if force_noise_std is not None:
        pixel_noise_std = force_noise_std
    else:
        pixel_noise_std = np.std(background_slice, axis=0) 
        
    err_signal_sum = pixel_noise_std * np.sqrt(n_signal_rows)
    err_bg_sum     = pixel_noise_std * np.sqrt(n_background_rows) * scale
    
    # Fehler des Netto-Signals
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
    # 1. Datenvorbereitung
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit = x[mask]
    y_fit = y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    if len(y_fit) < 5: 
        return None, None, None

    # --- Pre-Check: Ist da überhaupt Dynamik? ---
    # Wenn Max und Min fast gleich sind (< 0.05), ist es nur Rauschen.
    if (np.max(y_fit) - np.min(y_fit)) < 0.05:
        return None, None, None

    # --- Fit Setup ---
    # Wir zwingen den Fit, in der Mitte zu suchen, nicht am Rand
    window_width = fit_window[1] - fit_window[0]
    center_x = (fit_window[1] + fit_window[0]) / 2
    
    # Startwerte (Schätzung)
    A_guess  = np.max(y_fit) - np.median(y_fit)
    x0_guess = x_fit[np.argmax(y_fit)]
    s_guess  = window_width * 0.15 

    # Strenge Bounds: 
    # Sigma muss klein genug sein, um ein echter Peak zu sein (max 40% des Fensters)
    bounds = (
        (0, fit_window[0], 0.5), 
        (np.inf, fit_window[1], window_width * 0.4) 
    )

    try:
        popt, pcov = curve_fit(
            gaussian, x_fit, y_fit, p0=[A_guess, x0_guess, s_guess], 
            sigma=sigma_fit, absolute_sigma=True, 
            bounds=bounds, maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov)) # Die berechneten Fehler der Parameter
        
        amp_fit, mu_fit, sigma_fit_val = popt
        amp_err, mu_err, sigma_err     = perr

        # --- JETZT KOMMEN DIE STRENGEN FILTER ---

        # 1. "Unsicherheits-Killer" (Relative Error)
        # Wenn der Fehler der Amplitude mehr als 50% der Amplitude selbst ist, 
        # ist der Fit statistisches Raten. (Im Low Count ist amp_err oft riesig)
        if amp_err > 0.5 * amp_fit:
            return None, None, None
            
        # Dasselbe für die Breite: Wenn wir nicht wissen, wie breit er ist, ist er weg.
        if sigma_err > 0.5 * sigma_fit_val:
            return None, None, None

        # 2. "Phantom-Peak Killer" (SNR post-fit)
        # Wir berechnen das Rauschen der Datenpunkte UM den Fit herum (Residuen).
        residuals = y_fit - gaussian(x_fit, *popt)
        rmse_noise = np.std(residuals) # Root Mean Square Error

        # Das Signal (Amplitude) muss mindestens 3-mal stärker sein als das Restrauschen.
        # Im Low Count Bild ist das Signal vllt 0.2, aber das Rauschen auch 0.15 -> Ratio ~1.3 -> RAUS!
        if amp_fit < 3.0 * rmse_noise:
            return None, None, None

        # 3. "Positions-Check"
        # Wenn der Peak "irgendwo" ist (Fehler der Position > Breite des Peaks), ist er ungültig.
        if mu_err > sigma_fit_val:
            return None, None, None

        return gaussian(x, *popt), popt, perr

    except Exception:
        return None, None, None


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
        x_axis, signal, background, sbr, error, boxes = calculate_sbr_profiles(image, force_noise_std=noise_to_use)
        
        # Fit auf SBR
        fit_y, params, perr = perform_gaussian_fit(x_axis, sbr, y_err=error, fit_window=FIT_WINDOW_X)
        
        results.append({
            'x_axis':x_axis, 'signal':signal, 'background':background, 
            'sbr':sbr, 'error': error,
            'fit':fit_y, 'par':params, 'perr':perr, 'boxes':boxes
        })

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    for i in range(3):
        # Bild
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
        
        #  Berechnung der Höhen
        height_1 = r1_bottom - r1_top
        height_2 = r2_bottom - r2_top
        
        ax.add_patch(patches.Rectangle((ROI_X_START, r1_top), roi_w, height_1, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((ROI_X_START, r2_top), roi_w, height_2, lw=1, ec='red', fc='red', alpha=0.2))

        # Totale Intensitäten
        ax2 = axes[1, i]
        ax2.plot(results[i]['x_axis'], results[i]['signal'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(results[i]['x_axis'], results[i]['background'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(FIT_WINDOW_X[0], FIT_WINDOW_X[1], color='green', alpha=0.15, label='_Fit Region')
        ax2.grid(True, alpha=0.3)
        if i==0: 
            ax2.set_ylabel("Counts")
        if i==1: 
            ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(1.5, 6)

        # SRBR
        ax3 = axes[2, i]
        ax3.errorbar(results[i]['x_axis'], results[i]['sbr'], yerr=results[i]['error'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if i == 1: 
            # Prediction fit grün
            if results[1]['fit'] is not None:
                # parameters = [Amplitude, x0, sigma]
                amp_val = results[1]['par'][0]   # Amplitude
                mu_val  = results[1]['par'][1]   # Peak Position (x0)
                sigma_val = results[1]['par'][2] # Breite
                
                # Fehler
                mu_err = results[1]['perr'][1]
                sigma_err = results[1]['perr'][2]

                # Neuer Label-String mit Peak
                l = (f"Pred (Peak={mu_val:.1f}$\pm${mu_err:.1f}, "
                     f"$\sigma$={sigma_val:.2f}$\pm${sigma_err:.2f}, "
                     f"Max={np.max(results[1]['fit']):.2f})")
                     
                ax3.plot(results[1]['x_axis'], results[1]['fit'], color=FIT_COLORS[1], ls='--', lw=2.5, label=l)
        else: 
            # LC (i=0) und GT (i=2) Einzelplots
            if results[i]['fit'] is not None:
                amp_val = results[i]['par'][0]
                mu_val  = results[i]['par'][1]   # Peak Position
                sigma_val = results[i]['par'][2]
                
                mu_err = results[i]['perr'][1]
                sigma_err = results[i]['perr'][2]

                # Neuer Label-String mit Peak
                l = (f"Gauss (Peak={mu_val:.1f}$\pm${mu_err:.1f}, "
                     f"$\sigma$={sigma_val:.2f}$\pm${sigma_err:.2f}, "
                     f"Max={np.max(results[i]['fit']):.2f})")
                     
                ax3.plot(results[i]['x_axis'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
            
        ax3.set_xlabel("Pixel X")
        if i==0: 
            ax3.set_ylabel("SRBR: (Signal - Background) / Background")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlim(FIT_WINDOW_X)
        ax3.set_ylim(-0.1, 0.5)

    plt.tight_layout()
    out_name = f"Ana_{Path(NPZ_FILE).stem}_Slice{SLICE_INDEX}_SRBR_l.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot fertig: {out_name}")







# ---------------------------------------------------------
    # ZUSATZ: Checksummen für ALLE Slices der Serie
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f" CHECK: Intensitäts-Summen (Pixel Sum) für ALLE SLICES")
    print(f" Datei: {NPZ_FILE}")
    print("="*70)
    print(f"{'Slice':<6} | {'Low Count':<18} | {'Prediction':<18} | {'Ground Truth':<18}")
    print("-" * 70)

    # Zugriff auf die vollen Daten-Arrays aus dem geladenen 'data' Objekt
    # (Shape ist typischerweise: [Anzahl_Bilder, Höhe, Breite])
    full_lc   = data['lc']
    full_pred = data['pred']
    full_gt   = data['gt']
    
    # Sicherstellen, dass es eine Serie ist (3D Array)
    if full_lc.ndim == 3:
        n_slices = full_lc.shape[0]
        
        for i in range(n_slices):
            # Summen berechnen
            s_lc   = np.sum(full_lc[i])
            s_pred = np.sum(full_pred[i])
            s_gt   = np.sum(full_gt[i])
            
            # Kleiner Marker für das Bild, das wir oben geplottet haben
            marker = " <--- PLOTTED" if i == SLICE_INDEX else ""
            
            print(f"{i:<6} | {s_lc:<18.4f} | {s_pred:<18.4f} | {s_gt:<18.4f}{marker}")
            
    else:
        # Fallback, falls nur ein einzelnes Bild im File war
        s_lc   = np.sum(full_lc)
        s_pred = np.sum(full_pred)
        s_gt   = np.sum(full_gt)
        print(f"{0:<6} | {s_lc:<18.4f} | {s_pred:<18.4f} | {s_gt:<18.4f}")

    print("="*70 + "\n")

if __name__ == "__main__":
    main()