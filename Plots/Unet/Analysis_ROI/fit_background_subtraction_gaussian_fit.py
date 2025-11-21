#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path


# KONFIGURATION
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR   = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots/Unet/Analysis_ROI/Gaussian_fits"

NPZ_FILE = "Pred_unet_3d_SSIM_middle_improved_V2_D3_S12_Img19.npz"

# ROI settings
ROI_X = (0, 240)    
ROI_Y = (102, 117)  
BG_GAP        = 5 
BG_BOX_HEIGHT = 10  

FIT_WINDOW_X  = (20, 120) # Gauss fit Fenster

LINE_WIDTH_RAW = 1.0
LINE_WIDTH_NET = 1.0
PLOT_BINS      = 240

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Prediction", "Ground Truth"]


def vis_norm(image):
    """
    Percentile clipping für Visualisierung
    """
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    clipped_value = (np.clip(image, vmin, vmax) - vmin) / (vmax - vmin)
    return clipped_value

def bin_array(data, target_len):
    """
    Ändert die Länge eines Arrays auf 'target_len' um Auflösung des Plots zu steuern
    """
    old_positions = np.linspace(0, 1, len(data))
    new_positions = np.linspace(0, 1, target_len)
    resampled_data = np.interp(new_positions, old_positions, data)

    return resampled_data


def calculate_avg_profiles(img, x_range, y_range, bg_gap, bg_height, n_bins):
    x0, x1 = x_range; y0, y1 = y_range
    ty2, ty1 = max(0, y0-bg_gap), max(0, y0-bg_gap-bg_height)
    by1, by2 = min(img.shape[0], y1+bg_gap), min(img.shape[0], y1+bg_gap+bg_height)
    
    prof_sig = np.mean(img[y0:y1, x0:x1], axis=0)
    
    bg_list = []
    if ty2 > ty1: bg_list.append(img[ty1:ty2, x0:x1])
    if by2 > by1: bg_list.append(img[by1:by2, x0:x1])
    prof_bg = np.mean(np.concatenate(bg_list, axis=0), axis=0) if bg_list else np.zeros_like(prof_sig)

    prof_net = prof_sig - prof_bg
    
    if n_bins < len(prof_sig):
        prof_sig = bin_array(prof_sig, n_bins)
        prof_bg  = bin_array(prof_bg, n_bins)
        prof_net = bin_array(prof_net, n_bins)
        x = np.linspace(x0, x1, n_bins)
    else:
        x = np.arange(x0, x1)
    return x, prof_sig, prof_bg, prof_net, ((ty1, ty2), (by1, by2))

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_gaussian_fit(x, y, fit_window=None):
    # --- Daten maskieren ---
    if fit_window is not None:
        mask = (x >= fit_window[0]) & (x <= fit_window[1])
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y
    
    if len(x_fit) == 0: return None, None

    # Startwerte basierend auf dem Ausschnitt schätzen
    p0 = [np.max(y_fit)-np.min(y_fit), x_fit[np.argmax(y_fit)], (np.max(x_fit)-np.min(x_fit))*0.1]
    
    try:
        # Fit nur auf dem Ausschnitt (x_fit, y_fit)
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
        # Kurve aber für das GANZE x zurückgeben (zum Plotten)
        return gaussian(x, *popt), popt
    except: return None, None




def main():
    file_path = ROOT_DIR / "Plots/Unet/Analysis_ROI/Predictions_Raw" / NPZ_FILE
    data = np.load(file_path)
    images = [data['lc'], data['pred'], data['gt']] # images = [image_low_count, image_pred, image_ground_truth]
    
    results = []
    for i in images:
        x_axis, signal, background, pure_signal, boxes = calculate_avg_profiles(i, ROI_X, ROI_Y, BG_GAP, BG_BOX_HEIGHT, PLOT_BINS)
        
        # --- fit_window übergeben ---
        fit_y, params = perform_gaussian_fit(x_axis, pure_signal, fit_window=FIT_WINDOW_X)
        
        results.append({'x_axis':x_axis, 'signal':signal, 'background':background, 'pure_signal':pure_signal, 'fit':fit_y, 'par':params, 'boxes':boxes})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    for i in range(3):
        # 1. Bild
        ax = axes[0, i]
        ax.imshow(vis_norm(images[i]), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.axis('off')
        (ty1, ty2), (by1, by2) = results[i]['boxes']
        
        # Blaue Signal Box (Rahmen)
        ax.add_patch(patches.Rectangle((ROI_X[0], ROI_Y[0]), ROI_X[1]-ROI_X[0], ROI_Y[1]-ROI_Y[0], lw=2, ec='blue', fc='none'))
        
        # --- NEU: Grüne Fit-Region Box (gefüllt, transparent) ---
        # Liegt innerhalb der blauen Box, zeigt den genutzten X-Bereich
        fit_w = FIT_WINDOW_X[1] - FIT_WINDOW_X[0]
        fit_h = ROI_Y[1] - ROI_Y[0]
        if fit_w > 0:
            ax.add_patch(patches.Rectangle(
                (FIT_WINDOW_X[0], ROI_Y[0]), # xy (links oben)
                fit_w, fit_h,                # breite, höhe
                lw=0,                        # keine Randlinie, damit es nicht mit blau kollidiert
                fc='green',                  # Füllfarbe grün (passend zum Plot unten)
                alpha=0.2                    # Gleiche Transparenz wie rote Boxen
            ))
        # -------------------------------------------------------

        # Rote Background Boxen
        if ty2>ty1: ax.add_patch(patches.Rectangle((ROI_X[0], ty1), ROI_X[1]-ROI_X[0], ty2-ty1, lw=1, ec='red', fc='red', alpha=0.2))
        if by2>by1: ax.add_patch(patches.Rectangle((ROI_X[0], by1), ROI_X[1]-ROI_X[0], by2-by1, lw=1, ec='red', fc='red', alpha=0.2))

        # Rohe Intensitäten
        ax2 = axes[1, i]
        ax2.plot(results[i]['x_axis'], results[i]['signal'], color='blue', alpha=0.7, label='Raw Signal')
        ax2.plot(results[i]['x_axis'], results[i]['background'], color='red', alpha=0.7, label='Background')
        
        # --- Fit Bereich grün hinterlegen ---
        ax2.axvspan(FIT_WINDOW_X[0], FIT_WINDOW_X[1], color='green', alpha=0.1, label='Fit Region')
        
        ax2.grid(True, alpha=0.3)
        if i==0: ax2.set_ylabel("Avg Intensity")
        if i==1: ax2.legend(loc='upper right', fontsize=8)

        # Netto Intensitäten
        ax3 = axes[2, i]
        ax3.plot(results[i]['x_axis'], results[i]['pure_signal'], color='black', lw=LINE_WIDTH_NET, alpha=0.6, label='Net Data')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        # Fits
        if i == 1: # Mitte alle
            if results[0]['fit'] is not None: ax3.plot(results[0]['x_axis'], results[0]['fit'], color=FIT_COLORS[0], ls=':', lw=1.5)
            if results[2]['fit'] is not None: ax3.plot(results[2]['x_axis'], results[2]['fit'], color=FIT_COLORS[2], ls=':', lw=1.5)
            if results[1]['fit'] is not None: 
                l = f"Gauss ($\\sigma$={abs(results[1]['par'][2]):.1f})"
                ax3.plot(results[1]['x_axis'], results[1]['fit'], color=FIT_COLORS[1], ls='--', lw=2.5, label=l)
        else: # Einzeln
            if results[i]['fit'] is not None:
                l = f"Gauss ($\\sigma$={abs(results[i]['par'][2]):.1f})"
                ax3.plot(results[i]['x_axis'], results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)
            
        ax3.set_xlabel("Pixel X")
        if i==0: ax3.set_ylabel("Net Intensity")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    # Output Name basierend auf Input Name
    out_name = f"Ana_{Path(NPZ_FILE).stem}_ROI_Y{ROI_Y[0]}_FitRegionVis.png"
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot fertig: {out_name}")

if __name__ == "__main__":
    main()