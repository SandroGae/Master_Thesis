#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
import re
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. KONFIGURATION & PFADE
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Confidence")

# BEIDE Input-Ordner definieren
CARE_DIR  = BASE_DIR / "npz_files" / "CARE_10_SEEDS"
MIXED_DIR = BASE_DIR / "npz_files" / "Best_3_Points"

# Übergeordneter Output-Ordner
OUT_DIR_BASE = BASE_DIR / "Plots_L_Scan"

# Deine bewährte Serien-Konfiguration (L-Richtung / Horizontal)
SERIES_CONFIG = {
    # Block 1
    5:  {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    11: {"slice_idx": 19, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (40, 140),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    12: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (27, 127),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    13: {"slice_idx": 7,  "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 0, "bg_h": 10, "fit_window": (80, 180),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 88.0)},
    15: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (95, 195),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.5)},
    16: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (81, 181),  "y_lim_raw": (2.5, 7.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.5)},
    17: {"slice_idx": 17, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (136, 236), "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},    
    22: {"slice_idx": 16, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    29: {"slice_idx": 24, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (23, 123),  "y_lim_raw": (2.5, 5.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    30: {"slice_idx": 14, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (160, 240),   "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.0)},

    # Block 2
    32: {"slice_idx": 31, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (21, 121),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 92.0)},
    35: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (67, 167),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    36: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (21, 121),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    38: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (64, 164),  "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    41: {"slice_idx": 18, "roi_x": (0, 240), "roi_y": (98, 113),  "bg_gap": 0, "bg_h": 10, "fit_window": (0, 100),   "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.5)},
    42: {"slice_idx": 35, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    45: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (41, 141),  "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 97.0)},
    46: {"slice_idx": 37, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 90.0)},
    50: {"slice_idx": 12, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (59, 159),  "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
    51: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (108, 208), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.0)},

    # Block 3
    55: {"slice_idx": 21, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (122, 222), "y_lim_raw": (2.5, 4.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    56: {"slice_idx": 9,  "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (106, 206), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    57: {"slice_idx": 22, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (136, 236), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 98.0)},
    59: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (137, 237), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
    64: {"slice_idx": 5,  "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (140, 240), "y_lim_raw": (2.5, 7.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    67: {"slice_idx": 10, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (0, 100),   "y_lim_raw": (2.5, 6.5), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    68: {"slice_idx": 20, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 0, "bg_h": 10, "fit_window": (5, 105),   "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    72: {"slice_idx": 15, "roi_x": (0, 240), "roi_y": (102, 117), "bg_gap": 5, "bg_h": 10, "fit_window": (4, 104),   "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 94.0)},
    73: {"slice_idx": 25, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 5, "bg_h": 10, "fit_window": (42, 142), "y_lim_raw": (2.5, 6.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 95.0)},
    74: {"slice_idx": 23, "roi_x": (0, 240), "roi_y": (100, 115), "bg_gap": 0, "bg_h": 10, "fit_window": (32, 132), "y_lim_raw": (2.5, 5.0), "y_lim_sbr": (-0.1, 0.5), "vis_p": (0.5, 96.0)},
}

FIT_COLORS = ['darkorange', 'mediumseagreen', 'darkviolet']
TITLES     = ["Low Count", "Ensemble Reconstruction ($\mu_{ens}$)", "Ground Truth"]

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def vis_norm(image, p_low=0.5, p_high=99.5):
    vmin, vmax = np.percentile(image, [p_low, p_high])
    if vmax - vmin == 0: return image
    return np.clip((image - vmin) / (vmax - vmin), 0, 1)

def perform_gaussian_fit(x, y, y_err, fit_window):
    mask = (x >= fit_window[0]) & (x <= fit_window[1])
    x_f, y_f, s_f = x[mask], y[mask], y_err[mask]
    if len(y_f) < 5: return None, None, None
    win_w = fit_window[1] - fit_window[0]
    p0 = [np.max(y_f) - np.median(y_f), x_f[np.argmax(y_f)], win_w * 0.15]
    bounds = ((0, fit_window[0], 0.5), (np.inf, fit_window[1], win_w * 0.4))
    try:
        popt, pcov = curve_fit(gaussian, x_f, y_f, p0=p0, sigma=s_f, absolute_sigma=True, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return gaussian(x, *popt), popt, perr
    except:
        return None, None, None

# =====================================================
# 3. ENSEMBLE-PROZESS-FUNKTION
# =====================================================
def process_ensemble_combination(file_paths, s_id, cfg, out_dir, m_id):
    # Lade alle 10 Seeds und bilde den Mittelwert für mu
    mus = []
    lc_img, gt_img = None, None
    
    idx = cfg["slice_idx"]
    
    for k, path in enumerate(file_paths):
        data = np.load(path)
        # Slicing direkt beim Laden, um RAM zu sparen
        mus.append(data['pred'][idx])
        if k == 0:
            lc_img = data['lc'][idx]
            gt_img = data['gt'][idx]
            
    # Ensemble Average berechnen
    mu_ens = np.mean(mus, axis=0)

    # Die 3 Bilder für die Plots
    imgs = [lc_img, mu_ens, gt_img]

    rx, ry, bg_h = cfg["roi_x"], cfg["roi_y"], cfg["bg_h"]
    r1_t = max(0, ry[0] - cfg["bg_gap"] - bg_h); r1_b = r1_t + bg_h
    r2_t = min(192, ry[1] + cfg["bg_gap"]); r2_b = min(192, r2_t + bg_h)

    gt_bg_area = np.concatenate([imgs[2][r1_t:r1_b, rx[0]:rx[1]], imgs[2][r2_t:r2_b, rx[0]:rx[1]]])
    gt_std = np.std(gt_bg_area)

    results = []
    x_ax = np.arange(rx[0], rx[1])

    for i, img in enumerate(imgs):
        sig_s = img[ry[0]:ry[1], rx[0]:rx[1]]
        bg_s  = np.concatenate([img[r1_t:r1_b, rx[0]:rx[1]], img[r2_t:r2_b, rx[0]:rx[1]]])

        prof_sig = np.sum(sig_s, axis=0)
        scale = sig_s.shape[0] / bg_s.shape[0]
        prof_bg = np.sum(bg_s, axis=0) * scale

        denom = np.where(prof_bg == 0, 1e-9, prof_bg)
        sbr = (prof_sig - prof_bg) / denom
        p_std = (gt_std if i == 1 else np.std(bg_s)) + 1e-9

        n_sig_px = sig_s.shape[0] * sig_s.shape[1]
        n_bg_px = bg_s.shape[0] * bg_s.shape[1]
        err_net = np.sqrt((p_std * np.sqrt(n_sig_px/sig_s.shape[1]))**2 + (p_std * np.sqrt(n_bg_px/bg_s.shape[1]) * scale)**2)
        diff = np.where(prof_sig - prof_bg == 0, 1, prof_sig - prof_bg)
        sbr_err = np.abs(sbr) * np.sqrt((err_net/np.abs(diff))**2 + (p_std * np.sqrt(n_bg_px/bg_s.shape[1]) * scale / np.abs(denom))**2)

        fit_y, par, perr = (None, None, None) if i == 0 else perform_gaussian_fit(x_ax, sbr, sbr_err, cfg["fit_window"])
        results.append({'sig':prof_sig, 'bg':prof_bg, 'sbr':sbr, 'err':sbr_err, 'fit':fit_y, 'par':par, 'perr':perr})

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=100, gridspec_kw={'height_ratios': [1, 0.8, 1]})
    fig.suptitle(f"L-Scan (Horizontal) - {m_id} (Ensemble Average) - Serie {s_id}", fontsize=16, fontweight='bold')

    p_l, p_h = cfg.get("vis_p", (0.5, 99.5))
    for i in range(3):
        ax = axes[0, i]
        ax.imshow(vis_norm(imgs[i], p_l, p_h), cmap="gray_r")
        ax.set_title(TITLES[i], fontsize=14, fontweight='bold')
        roi_w, roi_h = rx[1]-rx[0], ry[1]-ry[0]
        ax.add_patch(patches.Rectangle((rx[0], ry[0]), roi_w, roi_h, lw=2, ec='blue', fc='none'))
        ax.add_patch(patches.Rectangle((rx[0], r1_t), roi_w, bg_h, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((rx[0], r2_t), roi_w, bg_h, lw=1, ec='red', fc='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((cfg["fit_window"][0], ry[0]), cfg["fit_window"][1]-cfg["fit_window"][0], roi_h, lw=0, fc='green', alpha=0.2))
        ax.axis('off')

        ax2 = axes[1, i]
        ax2.plot(x_ax, results[i]['sig'], color='blue', alpha=0.7, label='Raw Sum')
        ax2.plot(x_ax, results[i]['bg'], color='red', alpha=0.7, label='Background Sum')
        ax2.axvspan(cfg["fit_window"][0], cfg["fit_window"][1], color='green', alpha=0.1)
        ax2.set_ylim(cfg["y_lim_raw"])
        ax2.grid(True, alpha=0.3)
        if i == 0: ax2.set_ylabel("Counts")
        if i == 1: ax2.legend(loc='upper right', fontsize=8)

        ax3 = axes[2, i]
        ax3.errorbar(x_ax, results[i]['sbr'], yerr=results[i]['err'], fmt='.', markersize=5, color='black', alpha=0.6, label='SRBR')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)

        if results[i]['fit'] is not None:
            p, e = results[i]['par'], results[i]['perr']
            l = (f"Gauss (Amp={p[0]:.2f}$\pm${e[0]:.2f}, Peak={p[1]:.1f}$\pm${e[1]:.1f}, $\sigma$={p[2]:.2f})")
            ax3.plot(x_ax, results[i]['fit'], color=FIT_COLORS[i], ls='--', lw=2.5, label=l)

        ax3.set_xlim(cfg["fit_window"])
        ax3.set_ylim(cfg["y_lim_sbr"])
        ax3.set_xlabel("Pixel X")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_dir.mkdir(parents=True, exist_ok=True)
    # Speichern als übersichtliche Dateinamen
    fig.savefig(out_dir / f"L_Scan_S{s_id:02d}_{m_id}.png", bbox_inches='tight')
    plt.close(fig)

# =====================================================
# 4. RUNNER LOGIK (Gruppierung nach Modellen)
# =====================================================
def main():
    # Sammle Dateien aus BEIDEN Ordnern
    all_npzs = []
    if CARE_DIR.exists():
        all_npzs.extend(list(CARE_DIR.glob("*.npz")))
    if MIXED_DIR.exists():
        all_npzs.extend(list(MIXED_DIR.glob("*.npz")))
        
    ensembles = defaultdict(list)

    for f in all_npzs:
        s_id = None
        m_id = None
        
        # 1. Check auf P-Modelle (P02, P14, P23)
        match_p = re.search(r"(P\d+).*_S(\d+)\.npz", f.name)
        if match_p:
            m_id = match_p.group(1)
            s_id = int(match_p.group(2))
            
        # 2. Check auf CARE (Baseline)
        match_care = re.search(r"Confidence_MSE.*_S(\d+)\.npz", f.name)
        if match_care:
            m_id = "CARE_MSE"
            s_id = int(match_care.group(1))
            
        if m_id and s_id and s_id in SERIES_CONFIG:
            ensembles[(m_id, s_id)].append(f)

    if not ensembles:
        print("❌ Keine Dateien gefunden! Überprüfe die Pfade.")
        return

    print(f"Starte Ensemble-Plotting... (Insgesamt {len(ensembles)} Plots erwartet)")

    for (m_id, s_id), file_paths in ensembles.items():
        if len(file_paths) == 10:
            print(f" -> Erstelle Plot für {m_id} | Serie {s_id}...")
            model_out_dir = OUT_DIR_BASE / m_id
            process_ensemble_combination(file_paths, s_id, SERIES_CONFIG[s_id], model_out_dir, m_id)
        else:
            print(f"Warnung: Modell {m_id} Serie {s_id} hat nur {len(file_paths)} Seeds statt 10. Überspringe...")

    print("\n>>> Alle 120 L-Scan Plots wurden erfolgreich erstellt!")

if __name__ == "__main__":
    main()