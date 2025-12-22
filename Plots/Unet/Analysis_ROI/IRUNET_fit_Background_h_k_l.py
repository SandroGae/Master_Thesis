#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

# Unterdrücke Warnungen für saubereren Output
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# =====================================================
# Konfiguration & Pfade
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis")
IN_DIR   = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw"
OUT_DIR  = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Gaussian_fits_IRUNet"

# Datei Name
NPZ_FILE = "Pred_IRUNet_Series_12.npz"

# Globale Settings
TITLES = ["Low Count", "Prediction (IRUNet)", "Ground Truth"]
FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
SLICE_INDEX_FOR_PROFILES = 19 


def vis_norm(image):
    vmin, vmax = np.percentile(image, [0.5, 99.5])
    if vmax - vmin == 0: return image
    return (np.clip(image, vmin, vmax) - vmin) / (vmax - vmin)

def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_gaussian_fit(x, y, y_err, fit_window, mode='xy'):
    """
    Führt den Gauss-Fit durch.
    mode: 'z' nutzt die Heuristik aus deinem SRBR_h Skript.
          'xy' nutzt die Heuristik aus deinen SRBR_k/l Skripten.
    """
    # Maskierung auf den Fit-Bereich
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_fit = x[mask]
    y_fit = y[mask]
    sigma_fit = y_err[mask] if y_err is not None else None

    # Mindestens 4 Punkte für Fit nötig
    if np.sum(np.isfinite(y_fit)) < 4: return None, None, None
    
    # NaN Filter: Prüfe Y UND Sigma auf Gültigkeit
    valid = np.isfinite(y_fit)
    if sigma_fit is not None:
        valid = valid & np.isfinite(sigma_fit) & (sigma_fit > 0)
    
    x_fit, y_fit = x_fit[valid], y_fit[valid]
    if sigma_fit is not None: sigma_fit = sigma_fit[valid]
    
    if len(y_fit) < 4: return None, None, None 

    # --- UNTERSCHIEDLICHE HEURISTIK ---
    if mode == 'z':
        A  = np.max(y_fit)  # Nur Max
        s_factor = 0.2
    else:
        A  = np.max(y_fit) - np.min(y_fit) # Max - Min
        s_factor = 0.1

    if A == 0: A = 1.0 
    x0 = x_fit[np.argmax(y_fit)]
    s  = (np.max(x_fit) - np.min(x_fit)) * s_factor
    p0 = [A, x0, s]

    try:
        parameters, pcov = curve_fit(gaussian, x_fit, y_fit, p0, sigma=sigma_fit, absolute_sigma=True, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *parameters), parameters, perr
    except:
        return None, None, None

def plot_analysis(results, filename_suffix, roi_coords, axis_label, fit_window, images, plot_type):
    """
    Generische Plot Funktion
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coords

    # --- Globale Y-Limits für Counts (Reihe 2) berechnen ---
    all_vals = []
    for res in results:
        all_vals.append(res['signal'])
        all_vals.append(res['background'])
    all_vals_flat = np.concatenate(all_vals)
    
    y_min_counts = np.min(all_vals_flat)
    y_max_counts = np.max(all_vals_flat)
    
    # Margin für Counts-Plot
    y_range = y_max_counts - y_min_counts
    if y_range == 0: y_range = 1.0
    y_min_counts -= y_range * 0.1
    y_max_counts += y_range * 0.1

    for i in range(3):
        # -----------------------
        # Zeile 1: Bilder
        # -----------------------
        ax = axes[0, i]
        img_show = images[i] if plot_type != 'z' else images[i][SLICE_INDEX_FOR_PROFILES]
        
        ax.imshow(vis_norm(img_show), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        ax.axis('off')

        roi_w = roi_x_end - roi_x_start
        roi_h = roi_y_end - roi_y_start
        ax.add_patch(patches.Rectangle((roi_x_start, roi_y_start), roi_w, roi_h, lw=2, ec='blue', fc='none'))

        res = results[i]
        boxes = res.get('boxes')
        
        if plot_type == 'z': 
            ax.add_patch(patches.Rectangle((roi_x_start, roi_y_start), roi_w, roi_h, lw=0, fc='green', alpha=0.2))
            (t_y1, t_y2), (b_y1, b_y2) = boxes
            if (t_y2-t_y1) > 0: ax.add_patch(patches.Rectangle((roi_x_start, t_y1), roi_w, t_y2-t_y1, lw=1, ec='red', fc='red', alpha=0.2))
            if (b_y2-b_y1) > 0: ax.add_patch(patches.Rectangle((roi_x_start, b_y1), roi_w, b_y2-b_y1, lw=1, ec='red', fc='red', alpha=0.2))
            
        elif plot_type == 'y': 
            r_left, r_right = boxes
            bg_w = r_right - r_left
            if bg_w > 0: ax.add_patch(patches.Rectangle((r_left, roi_y_start), bg_w, roi_h, lw=1, ec='red', fc='red', alpha=0.2))
            fit_h = fit_window[1] - fit_window[0]
            ax.add_patch(patches.Rectangle((roi_x_start, fit_window[0]), roi_w, fit_h, lw=0, fc='green', alpha=0.2))

        elif plot_type == 'x': 
            (r1_t, r1_b), (r2_t, r2_b) = boxes
            if (r1_b-r1_t) > 0: ax.add_patch(patches.Rectangle((roi_x_start, r1_t), roi_w, r1_b-r1_t, lw=1, ec='red', fc='red', alpha=0.2))
            if (r2_b-r2_t) > 0: ax.add_patch(patches.Rectangle((roi_x_start, r2_t), roi_w, r2_b-r2_t, lw=1, ec='red', fc='red', alpha=0.2))
            fit_w = fit_window[1] - fit_window[0]
            ax.add_patch(patches.Rectangle((fit_window[0], roi_y_start), fit_w, roi_h, lw=0, fc='green', alpha=0.2))

        # -----------------------
        # Zeile 2: Counts
        # -----------------------
        ax2 = axes[1, i]
        ax2.plot(res['x_axis'], res['signal'], color='blue', alpha=0.7, label='Signal')
        ax2.plot(res['x_axis'], res['background'], color='red', alpha=0.7, label='Background')
        ax2.axvspan(fit_window[0], fit_window[1], color='green', alpha=0.15, label='Fit Region')
        ax2.grid(True, alpha=0.3)
        if i==0: ax2.set_ylabel("Counts")
        if i==1: ax2.legend(loc='upper right', fontsize=8)
        ax2.set_ylim(y_min_counts, y_max_counts)

        # -----------------------
        # Zeile 3: SRBR & Fit
        # -----------------------
        ax3 = axes[2, i]
        ax3.errorbar(res['x_axis'], res['sbr'], yerr=res['error'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        
        if res['fit'] is not None:
            mu, mu_err = res['par'][1], res['perr'][1]
            sigma, sigma_err = res['par'][2], res['perr'][2]
            
            # --- MAX HINZUFÜGEN ---
            max_val = np.max(res['fit'])
            
            label = (f"Fit (Peak={mu:.1f}$\pm${mu_err:.1f}, "
                     f"$\sigma$={sigma:.2f}$\pm${sigma_err:.2f}, "
                     f"Max={max_val:.2f})")
            
            ax3.plot(res['x_axis'], res['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=label)
        
        # --- HIER IST DIE ÄNDERUNG (Variable axis_label wird benutzt) ---
        ax3.set_xlabel(axis_label)
        
        if i==0: ax3.set_ylabel("SRBR")
        ax3.set_ylim(-0.1, 0.5) 
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)
        
        # X-Limit auf Fit-Window
        ax3.set_xlim(fit_window[0], fit_window[1])

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{filename_suffix}.png"
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)
    print(f"Plot erstellt: {out_name}")

# =====================================================
# 2. Analyse Logik: Z-Achse (SRBR_h)
# =====================================================
def analyze_z_axis(volumes):
    ROI_X_START, ROI_X_END = 65, 86
    ROI_Y_START, ROI_Y_END = 104, 115
    BG_GAP, BG_HEIGHT      = 5, 10
    FIT_WINDOW             = (2, 38)
    
    bg_t_b = max(0, ROI_Y_START - BG_GAP)
    bg_t_t = max(0, bg_t_b - BG_HEIGHT)
    bg_b_t = ROI_Y_END + BG_GAP
    bg_b_b = bg_b_t + BG_HEIGHT
    n_sig_pix = (ROI_X_END - ROI_X_START) * (ROI_Y_END - ROI_Y_START)
    
    gt_vol = volumes[2]
    gt_noise_std = []
    for img in gt_vol:
        bg_pix = []
        if bg_t_b > bg_t_t: bg_pix.append(img[bg_t_t:bg_t_b, ROI_X_START:ROI_X_END])
        if bg_b_b > bg_b_t and bg_b_b <= img.shape[0]: bg_pix.append(img[bg_b_t:bg_b_b, ROI_X_START:ROI_X_END])
        if bg_pix: gt_noise_std.append(np.std(np.concatenate(bg_pix)))
        else: gt_noise_std.append(1e-9)
    gt_noise_std = np.array(gt_noise_std)

    results = []
    for idx, vol in enumerate(volumes):
        n_frames = vol.shape[0]
        p_sig, p_bg, p_sbr, p_err = [], [], [], []
        noise_src = gt_noise_std if idx == 1 else None 

        for i in range(n_frames):
            img = vol[i]
            sig = np.sum(img[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END])
            
            bg_pix_list = []
            if bg_t_b > bg_t_t: bg_pix_list.append(img[bg_t_t:bg_t_b, ROI_X_START:ROI_X_END])
            if bg_b_b > bg_b_t and bg_b_b <= img.shape[0]: bg_pix_list.append(img[bg_b_t:bg_b_b, ROI_X_START:ROI_X_END])
            
            if bg_pix_list:
                bg_concat = np.concatenate(bg_pix_list)
                mean_bg = np.mean(bg_concat)
                sum_bg_eq = mean_bg * n_sig_pix
                n_bg_pix = bg_concat.size
                scale = n_sig_pix / n_bg_pix
                
                net = sig - sum_bg_eq
                denom = sum_bg_eq if sum_bg_eq > 1e-9 else 1e-9
                sbr = net / denom
                
                std = noise_src[i] if noise_src is not None else np.std(bg_concat)
                err_sig = std * np.sqrt(n_sig_pix)
                err_bg  = std * np.sqrt(n_bg_pix) * scale
                err_net = np.sqrt(err_sig**2 + err_bg**2)
                
                rel_err = np.sqrt((err_net/abs(net if abs(net)>1e-9 else 1))**2 + (err_bg/abs(denom))**2)
                err_val = abs(sbr) * rel_err
            else:
                sum_bg_eq, sbr, err_val = 0, 0, 0

            p_sig.append(sig); p_bg.append(sum_bg_eq); p_sbr.append(sbr); p_err.append(err_val)
        
        x_axis = np.arange(n_frames)
        fit_y, par, perr = perform_gaussian_fit(x_axis, np.array(p_sbr), np.array(p_err), FIT_WINDOW, mode='z')
        
        results.append({
            'x_axis': x_axis, 'signal': np.array(p_sig), 'background': np.array(p_bg),
            'sbr': np.array(p_sbr), 'error': np.array(p_err),
            'fit': fit_y, 'par': par, 'perr': perr,
            'boxes': ((bg_t_t, bg_t_b), (bg_b_t, bg_b_b))
        })

    # --- ÄNDERUNG HIER: "Image Index (Z-Axis)" ---
    plot_analysis(results, "IRUNet_Ana_SRBR_h", (ROI_X_START, ROI_X_END, ROI_Y_START, ROI_Y_END), "Image Index (Z-Axis)", FIT_WINDOW, volumes, 'z')

# =====================================================
# 3. Analyse Logik: Y-Achse / k (SRBR_k)
# =====================================================
def analyze_y_axis(images):
    ROI_X_START, ROI_X_END = 60, 81
    ROI_Y_START, ROI_Y_END = 0, 192
    FIT_WINDOW             = (90, 130)
    
    bg_left  = min(240, ROI_X_END + 80)
    bg_right = min(240, bg_left + 20)
    gt_img = images[2]
    bg_slice = gt_img[ROI_Y_START:ROI_Y_END, bg_left:bg_right]
    gt_std = np.std(bg_slice)

    results = []
    for idx, img in enumerate(images):
        sig_slice = img[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        bg_slice  = img[ROI_Y_START:ROI_Y_END, bg_left:bg_right]
        
        prof_sig = np.sum(sig_slice, axis=1)
        prof_bg_raw = np.sum(bg_slice, axis=1)
        scale = sig_slice.shape[1] / bg_slice.shape[1] if bg_slice.shape[1] > 0 else 0
        prof_bg = prof_bg_raw * scale
        
        net = prof_sig - prof_bg
        denom = prof_bg.copy(); denom[denom==0] = 1e-9
        sbr = net / denom
        
        std = gt_std if idx == 1 else np.std(bg_slice)
        err_sig = std * np.sqrt(sig_slice.shape[1])
        err_bg  = std * np.sqrt(bg_slice.shape[1]) * scale
        err_net = np.sqrt(err_sig**2 + err_bg**2)
        
        # --- FIX: Safe division for arrays ---
        safe_net = np.abs(net)
        safe_net[safe_net < 1e-9] = 1.0 
        safe_denom = np.abs(denom)
        
        rel_err = np.sqrt((err_net/safe_net)**2 + (err_bg/safe_denom)**2)
        error = np.abs(sbr) * rel_err
        
        # Säubere NaNs
        error[~np.isfinite(error)] = 1e-9

        x_axis = np.arange(ROI_Y_START, ROI_Y_END)
        fit_y, par, perr = perform_gaussian_fit(x_axis, sbr, error, FIT_WINDOW, mode='xy')
        
        results.append({
            'x_axis': x_axis, 'signal': prof_sig, 'background': prof_bg,
            'sbr': sbr, 'error': error,
            'fit': fit_y, 'par': par, 'perr': perr,
            'boxes': (bg_left, bg_right)
        })

    # --- ÄNDERUNG HIER: "Pixel Y" ---
    plot_analysis(results, "IRUNet_Ana_SRBR_k", (ROI_X_START, ROI_X_END, ROI_Y_START, ROI_Y_END), "Pixel Y", FIT_WINDOW, images, 'y')

# =====================================================
# 4. Analyse Logik: X-Achse / l (SRBR_l)
# =====================================================
def analyze_x_axis(images):
    ROI_X_START, ROI_X_END = 0, 240
    ROI_Y_START, ROI_Y_END = 102, 117
    BG_GAP, BG_HEIGHT      = 5, 10
    FIT_WINDOW             = (20, 120)

    r1_b = max(0, ROI_Y_START - BG_GAP)
    r1_t = max(0, r1_b - BG_HEIGHT)
    r2_t = ROI_Y_END + BG_GAP
    r2_b = r2_t + BG_HEIGHT

    gt_img = images[2]
    bg_list = []
    bg_list.append(gt_img[r1_t:r1_b, ROI_X_START:ROI_X_END])
    bg_list.append(gt_img[r2_t:r2_b, ROI_X_START:ROI_X_END])
    gt_std_arr = np.std(np.concatenate(bg_list, axis=0), axis=0)

    results = []
    for idx, img in enumerate(images):
        sig_slice = img[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
        prof_sig = np.sum(sig_slice, axis=0)
        
        bg_l = []
        bg_l.append(img[r1_t:r1_b, ROI_X_START:ROI_X_END])
        bg_l.append(img[r2_t:r2_b, ROI_X_START:ROI_X_END])
        bg_slice = np.concatenate(bg_l, axis=0)
        
        prof_bg_raw = np.sum(bg_slice, axis=0)
        scale = sig_slice.shape[0] / bg_slice.shape[0] if bg_slice.shape[0] > 0 else 0
        prof_bg = prof_bg_raw * scale
        
        net = prof_sig - prof_bg
        denom = prof_bg.copy(); denom[denom==0] = 1e-9
        sbr = net / denom
        
        std = gt_std_arr if idx == 1 else np.std(bg_slice, axis=0)
        err_sig = std * np.sqrt(sig_slice.shape[0])
        err_bg  = std * np.sqrt(bg_slice.shape[0]) * scale
        
        # --- FIX: Safe division for arrays ---
        safe_net = np.abs(net)
        safe_net[safe_net < 1e-9] = 1.0 
        safe_denom = np.abs(denom)

        rel_err = np.sqrt(((np.sqrt(err_sig**2 + err_bg**2))/safe_net)**2 + (err_bg/safe_denom)**2)
        error = np.abs(sbr) * rel_err
        
        # Säubere NaNs
        error[~np.isfinite(error)] = 1e-9
        
        x_axis = np.arange(ROI_X_START, ROI_X_END)
        fit_y, par, perr = perform_gaussian_fit(x_axis, sbr, error, FIT_WINDOW, mode='xy')
        
        results.append({
            'x_axis': x_axis, 'signal': prof_sig, 'background': prof_bg,
            'sbr': sbr, 'error': error,
            'fit': fit_y, 'par': par, 'perr': perr,
            'boxes': ((r1_t, r1_b), (r2_t, r2_b))
        })
    
    # --- ÄNDERUNG HIER: "Pixel X" ---
    plot_analysis(results, "IRUNet_Ana_SRBR_l", (ROI_X_START, ROI_X_END, ROI_Y_START, ROI_Y_END), "Pixel X", FIT_WINDOW, images, 'x')

# =====================================================
# Main
# =====================================================
def main():
    file_path = IN_DIR / NPZ_FILE
    if not file_path.exists():
        print(f"Datei nicht gefunden: {file_path}")
        return

    print(f"Lade Daten: {file_path.name}")
    data = np.load(file_path)
    lc, pred, gt = data['lc'], data['pred'], data['gt']
    volumes = [lc, pred, gt]

    print("--- 1. Analyse: Z-Achse (SRBR_h) ---")
    analyze_z_axis(volumes)

    print(f"--- 2. Analyse: Y-Achse (SRBR_k) für Slice {SLICE_INDEX_FOR_PROFILES} ---")
    images_slice = [v[SLICE_INDEX_FOR_PROFILES] for v in volumes]
    analyze_y_axis(images_slice)

    print(f"--- 3. Analyse: X-Achse (SRBR_l) für Slice {SLICE_INDEX_FOR_PROFILES} ---")
    analyze_x_axis(images_slice)

    print("\nAlle Analysen abgeschlossen.")

if __name__ == "__main__":
    main()