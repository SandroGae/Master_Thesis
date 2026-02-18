import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless-Modus für Cluster
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
# 3. DATEN-LOGIK (Scan + Duplizierung für Quadrat)
# =====================================================
def collect_data():
    results = []
    all_jsons = list(MODELS_ROOT.glob("Point_*/*.json"))
    print(f"Scanne {len(all_jsons)} JSON-Dateien und NPZs...")

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
            p_gt = np.sum(data['gt'][15], axis=1) 
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

    # --- SPEZIAL-LOGIK: Punkt 43 duplizieren für Alpha=1 Kante ---
    # Wir suchen den Punkt, der Alpha=1 repräsentiert (normalerweise Point_43)
    p43_data = df[df['alpha'] >= 0.99]
    if not p43_data.empty:
        print("Dupliziere Punkt 43 Daten für die gesamte Alpha=1 Kante (Quadrat-Fix)...")
        # Wir erstellen 7 künstliche Beta-Werte (0.0 bis 1.0) für Alpha=1
        beta_grid = np.linspace(0, 1, 7)
        extra_rows = []
        for b_val in beta_grid:
            # Nehme alle 10 Seeds von Punkt 43 und setze sie auf die neue Beta-Position
            for _, row in p43_data.iterrows():
                new_row = row.copy()
                new_row['beta'] = b_val
                new_row['alpha'] = 1.0
                extra_rows.append(new_row)
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    return df

# =====================================================
# 4. PLOTTING (1x3 und 1x2 Layout)
# =====================================================
def plot_master(df):
    # Aggregation (wie gehabt)
    df_h = df[df['aborted'] == False].groupby(['alpha', 'beta']).mean().reset_index()
    df_a = df.groupby(['alpha', 'beta']).mean().reset_index()
    df_s = df.groupby(['alpha', 'beta']).apply(lambda x: 1 - x['aborted'].mean()).reset_index(name='stab')

    # Grid Interpolation
    ai, bi = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    grid_a, grid_b = np.meshgrid(ai, bi)
    
    def get_zi(d, col):
        return griddata((d['alpha'], d['beta']), d[col], (grid_a, grid_b), method='linear')

    fig = plt.figure(figsize=(22, 12), dpi=150)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

    # --- Zeile 1: Area & Stabilität (1x3) ---
    plots_r1 = [
        (df_h, 'area', 'Area Ratio (Healthy)', 'plasma'),
        (df_a, 'area', 'Area Ratio (All Seeds)', 'plasma'),
        (df_s, 'stab', 'Training Stability', 'RdYlGn')
    ]

    for i, (data, col, title, cmap) in enumerate(plots_r1):
        ax = fig.add_subplot(gs[0, i])
        zi = get_zi(data, col)
        cf = ax.contourf(grid_a, grid_b, zi, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Y-Label nur ganz links (Spalte 0)
        if i == 0:
            ax.set_ylabel("Beta (MAE/MSE)", fontsize=12)
        
        # X-Label nur, wenn kein Plot mehr darunter kommt (hier nur für Stability Spalte 2)
        if i == 2:
            ax.set_xlabel("Alpha (SSIM)", fontsize=12)

    # --- Zeile 2: Mu-Shift (1x2) ---
    plots_r2 = [
        (df_h, 'shift', 'Mu-Shift (Healthy)', 'magma'),
        (df_a, 'shift', 'Mu-Shift (All Seeds)', 'magma')
    ]

    for i, (data, col, title, cmap) in enumerate(plots_r2):
        ax = fig.add_subplot(gs[1, i])
        zi = get_zi(data, col)
        cf = ax.contourf(grid_a, grid_b, zi, levels=50, cmap=cmap)
        plt.colorbar(cf, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Y-Label nur ganz links (Spalte 0)
        if i == 0:
            ax.set_ylabel("Beta (MAE/MSE)", fontsize=12)
        
        # X-Label unter beiden Plots der untersten Zeile
        ax.set_xlabel("Alpha (SSIM)", fontsize=12)

    # Info-Box (Spalte 2, Zeile 2)
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis('off')

    plt.tight_layout()
    save_path = OUT_DIR / "Master_Heatmap_Comparison_Clean.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot gespeichert: {save_path}")
    
# =====================================================
# 5. MAIN LOGIK (mit CSV-Check)
# =====================================================
def main():
    if CSV_PATH.exists():
        print(f"Lade existierende Daten aus CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        print("Erstelle neue Datenbank aus JSONs/NPZs...")
        df = collect_data()
        df.to_csv(CSV_PATH, index=False)
        print(f"Datenbank gespeichert unter: {CSV_PATH}")

    plot_master(df)

if __name__ == "__main__":
    main()