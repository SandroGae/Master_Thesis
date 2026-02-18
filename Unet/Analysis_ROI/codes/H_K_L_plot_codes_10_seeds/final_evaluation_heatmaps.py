import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib.gridspec as gridspec

# =====================================================
# 1. SETUP & PFADE
# =====================================================
MODELS_ROOT = Path.home() / "scratch" / "43_Models_10_Seeds"
IN_DIR      = Path.home() / "scratch" / "Evaluation_Pipeline" / "Evaluation_results"
OUT_DIR     = Path.home() / "scratch" / "Evaluation_Pipeline" / "Final_Heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_IDS = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50]

# =====================================================
# 2. PHYSIK-KERN (Gauß-Fit)
# =====================================================
def gaussian(x, a, mu, sigma): 
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_fit(x, y, win=[90, 130]):
    mask = (x >= win[0]) & (x <= win[1])
    xf, yf = x[mask], y[mask]
    if len(yf) < 5 or np.max(yf) <= 1e-7: return None
    p0 = [np.max(yf), xf[np.argmax(yf)], 5.0]
    try:
        popt, _ = curve_fit(gaussian, xf, yf, p0=p0, 
                           bounds=([0, win[0], 0.5], [np.inf, win[1], 15.0]), maxfev=2000)
        return {"area": popt[0] * popt[2], "mu": popt[1]}
    except: return None

# =====================================================
# 3. DATEN-SAMMLUNG
# =====================================================
def collect_data():
    results = []
    all_jsons = list(MODELS_ROOT.glob("Point_*/*.json"))
    print(f"Scanne {len(all_jsons)} JSON-Dateien...")

    for j_path in all_jsons:
        with open(j_path, 'r') as f:
            meta = json.load(f)
        
        alpha, beta, seed = meta['alpha'], meta['beta'], meta['seed']
        aborted = meta.get('aborted', False)
        model_id = j_path.stem
        
        area_ratios, shifts = [], []
        for s_id in SERIES_IDS:
            npz_p = IN_DIR / f"Eval_{model_id}_S{s_id}.npz"
            if not npz_p.exists(): continue
            
            data = np.load(npz_p)
            # Wir nehmen den mittleren Slice der NPZ (K-Richtung Profil)
            p_gt = np.sum(data['gt'][15], axis=1) # Beispiel Slice 15
            p_pr = np.sum(data['pred'][15], axis=1)
            x = np.arange(len(p_gt))
            
            f_gt, f_pr = perform_fit(x, p_gt), perform_fit(x, p_pr)
            if f_gt and f_pr:
                area_ratios.append(f_pr["area"] / f_gt["area"])
                shifts.append(abs(f_pr["mu"] - f_gt["mu"]))

        if area_ratios:
            results.append({
                "alpha": alpha, "beta": beta, "seed": seed, "aborted": aborted,
                "area": np.mean(area_ratios), "shift": np.mean(shifts)
            })
    return pd.DataFrame(results)

# =====================================================
# 4. HEATMAP GENERATOR
# =====================================================
def plot_master(df):
    # Aggregation der 3 Datensätze
    df_h = df[df['aborted'] == False].groupby(['alpha', 'beta']).mean().reset_index()
    df_a = df.groupby(['alpha', 'beta']).mean().reset_index()
    df_s = df.groupby(['alpha', 'beta']).apply(lambda x: 1 - x['aborted'].mean()).reset_index(name='stab')

    # Grid Interpolation
    ai, bi = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    grid_a, grid_b = np.meshgrid(ai, bi)
    
    def get_zi(d, col):
        return griddata((d['alpha'], d['beta']), d[col], (grid_a, grid_b), method='cubic')

    # Layout: Zeile 1 (1x3) für Area/Stab, Zeile 2 (1x2) für Shift
    fig = plt.figure(figsize=(20, 12), dpi=150)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # --- Zeile 1: Area & Stabilität ---
    titles_r1 = ["Area Ratio (Healthy)", "Area Ratio (All Seeds)", "Training Stability"]
    data_r1 = [(df_h, 'area', 'plasma'), (df_a, 'area', 'plasma'), (df_s, 'stab', 'RdYlGn')]

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        zi = get_zi(data_r1[i][0], data_r1[i][1])
        cf = ax.contourf(grid_a, grid_b, zi, levels=50, cmap=data_r1[i][2])
        fig.colorbar(cf, ax=ax)
        ax.scatter(data_r1[i][0]['alpha'], data_r1[i][0]['beta'], c='white', s=10, alpha=0.3)
        ax.set_title(titles_r1[i], fontweight='bold')

    # --- Zeile 2: Mu-Shift ---
    titles_r2 = ["Mu-Shift (Healthy)", "Mu-Shift (All Seeds)"]
    data_r2 = [(df_h, 'shift', 'magma'), (df_a, 'shift', 'magma')]

    for i in range(2):
        ax = fig.add_subplot(gs[1, i])
        zi = get_zi(data_r2[i][0], data_r2[i][1])
        cf = ax.contourf(grid_a, grid_b, zi, levels=50, cmap=data_r2[i][2])
        fig.colorbar(cf, ax=ax)
        ax.set_title(titles_r2[i], fontweight='bold')

    # Labels für alle
    for ax in fig.axes:
        ax.set_xlabel("Alpha (SSIM)")
        ax.set_ylabel("Beta (MAE/MSE)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Master_Heatmap_Comparison.png")
    
    # Text-Zusammenfassung
    with open(OUT_DIR / "final_stats.txt", "w") as f:
        f.write(f"Gesamt-Auswertung von {len(df)} Einzelläufen.\n")
        f.write(f"Bester Healthy Area-Ratio: {df_h['area'].max():.4f}\n")
        f.write(f"Niedrigster Healthy Shift: {df_h['shift'].min():.4f} px\n")

if __name__ == "__main__":
    data_df = collect_data()
    plot_master(data_df)
    print(f"Erfolg! Check den Ordner: {OUT_DIR}")