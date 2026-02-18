import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Wichtig für Cluster
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
CSV_PATH    = OUT_DIR / "evaluation_metrics_database.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_IDS = [5, 11, 12, 15, 16, 21, 22, 29, 35, 50]
BETA_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

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
# 3. DATEN-SAMMLUNG (mit Alpha=1 Expansion)
# =====================================================
def collect_data():
    results = []
    all_jsons = list(MODELS_ROOT.glob("Point_*/*.json"))
    print(f"Scanne {len(all_jsons)} Dateien...")

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
            p_gt = np.sum(data['gt'][15], axis=1) # Slice 15 als Referenz
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
    
    df = pd.DataFrame(results)

    # --- ALPHA=1 EXPANSION (Quadrat-Logik) ---
    a1_data = df[df['alpha'] >= 0.99].copy()
    if not a1_data.empty:
        print("Expanding Alpha=1.0 boundary for square topology...")
        extra_rows = []
        for b_val in BETA_STEPS:
            if b_val == a1_data['beta'].iloc[0]: continue
            temp = a1_data.copy()
            temp['beta'] = b_val
            temp['alpha'] = 1.0
            extra_rows.append(temp)
        df = pd.concat([df] + extra_rows, ignore_index=True)

    return df

# =====================================================
# 4. PLOTTING (Analog zum Referenz-Design)
# =====================================================
def plot_master(df):
    # Aggregation
    df_h = df[df['aborted'] == False].groupby(['alpha', 'beta']).mean().reset_index()
    df_a = df.groupby(['alpha', 'beta']).mean().reset_index()
    df_s = df.groupby(['alpha', 'beta']).apply(lambda x: 1 - x['aborted'].mean(), include_groups=False).reset_index(name='stab')

    # Grid Interpolation
    xi, yi = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    
    def draw_heatmap(ax, d, col, title, cmap, show_y=False, show_x=False):
        zi = griddata((d['alpha'], d['beta']), d[col], (xi, yi), method='linear')
        cf = ax.contourf(xi, yi, zi, levels=50, cmap=cmap)
        ax.contour(xi, yi, zi, levels=15, colors='white', alpha=0.3)
        ax.scatter(d['alpha'], d['beta'], c='white', edgecolors='black', s=20, alpha=0.5)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=14)
        if show_y: ax.set_ylabel("Beta (MAE/MSE)", fontsize=12)
        if show_x: ax.set_xlabel("Alpha (SSIM)", fontsize=12)

    fig = plt.figure(figsize=(24, 12), dpi=150)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Zeile 1: Area & Stability (1x3)
    draw_heatmap(fig.add_subplot(gs[0, 0]), df_h, 'area', 'Area Ratio (Healthy)', 'plasma', show_y=True)
    draw_heatmap(fig.add_subplot(gs[0, 1]), df_a, 'area', 'Area Ratio (All Seeds)', 'plasma')
    draw_heatmap(fig.add_subplot(gs[0, 2]), df_s, 'stab', 'Training Stability', 'RdYlGn', show_x=True)

    # Zeile 2: Mu-Shift (1x2)
    draw_heatmap(fig.add_subplot(gs[1, 0]), df_h, 'shift', 'Mu-Shift (Healthy)', 'magma', show_y=True, show_x=True)
    draw_heatmap(fig.add_subplot(gs[1, 1]), df_a, 'shift', 'Mu-Shift (All Seeds)', 'magma', show_x=True)

    plt.tight_layout()
    save_p = OUT_DIR / "Master_Heatmap_Analog_Design.png"
    plt.savefig(save_p, bbox_inches='tight')
    print(f"Plot erfolgreich gespeichert: {save_p}")

def main():
    if CSV_PATH.exists():
        print(f"Lade Daten aus Cache: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        df = collect_data()
        df.to_csv(CSV_PATH, index=False)
    
    plot_master(df)

if __name__ == "__main__":
    main()