import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# --- PFADE ANPASSEN ---
PARENT_CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV")
FIG_BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Figures")

# --- EINSTELLUNGEN ---
CHOSEN_CMAP = 'plasma'

def get_folder_selection():
    subdirs = sorted([d.name for d in PARENT_CSV_DIR.iterdir() if d.is_dir()])
    print("\nWelchen Ordner für die Success-Score Heatmap verwenden?")
    for i, d in enumerate(subdirs):
        print(f"[{i}] {d}")
    idx = int(input("\nAuswahl: "))
    return subdirs[idx]

def extract_composite_scores(folder_path):
    results = []
    pattern = re.compile(r"a(\d+\.\d+)_b(\d+\.\d+)")
    
    # 1. Schritt: Alle Daten sammeln für globale Normalisierung
    raw_data = []
    for csv_file in folder_path.glob("log_*.csv"):
        match = pattern.search(csv_file.name)
        if match:
            df = pd.read_csv(csv_file)
            # Wir nehmen den Punkt des niedrigsten val_loss (wie besprochen)
            best_idx = df['val_loss'].idxmin()
            best_row = df.loc[best_idx]
            
            raw_data.append({
                'alpha': float(match.group(1)),
                'beta': float(match.group(2)),
                'mae': best_row['val_mae_center'],
                'mse': best_row['val_mse_center'],
                'ssim': best_row['val_ssim_center']
            })
    
    if not raw_data: return pd.DataFrame()
    df_raw = pd.DataFrame(raw_data)

    # 2. Schritt: Normalisierung (Min-Max)
    def normalize(series, reverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min: return series * 0 + 1.0
        return (s_max - series) / (s_max - s_min) if reverse else (series - s_min) / (s_max - s_min)

    df_raw['n_mae'] = normalize(df_raw['mae'], reverse=True)
    df_raw['n_mse'] = normalize(df_raw['mse'], reverse=True)
    df_raw['n_ssim'] = normalize(df_raw['ssim'], reverse=False)

    # 3. Schritt: Success Score berechnen
    df_raw['score'] = (df_raw['n_mae'] + df_raw['n_mse'] + df_raw['n_ssim']) / 3.0
    return df_raw

def plot_heatmaps(df, folder_name):
    target_dir = FIG_BASE_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Interpolations-Grid
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(0, 1, 200)
    xi, yi = np.meshgrid(xi, yi)
    # Wir plotten jetzt 'score' statt 'psnr'
    zi = griddata((df['alpha'], df['beta']), df['score'], (xi, yi), method='cubic')

    # --- PLOT 1: 2D Heatmap ---
    fig_2d = plt.figure(figsize=(12, 9))
    cp = plt.contourf(xi, yi, zi, levels=60, cmap=CHOSEN_CMAP)
    cbar = plt.colorbar(cp)
    cbar.set_label('Normalized Composite Score (NCS)', fontsize=14, labelpad=15)

    plt.scatter(df['alpha'], df['beta'], c='white', s=40, edgecolors='black', alpha=0.8, label='Runs')
    contours = plt.contour(xi, yi, zi, levels=15, colors='white', alpha=0.3)
    plt.clabel(contours, inline=True, fontsize=9, fmt='%.2f')

    plt.title(f'Success Score Heatmap', fontsize=16, pad=20)
    plt.xlabel('Alpha (SSIM)', fontsize=14)
    plt.ylabel('Beta (MSE/MAE)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(target_dir / "success_score_2d.png", dpi=300)
    plt.close()

    # --- PLOT 2: 3D Surface Plot ---
    fig_3d = plt.figure(figsize=(14, 10))
    ax = fig_3d.add_subplot(111, projection='3d')
    ls = LightSource(azdeg=315, altdeg=45)
    
    # zi_clipped gegen NaNs absichern falls Interpolation am Rand wegläuft
    zi_clean = np.nan_to_num(zi, nan=np.nanmin(zi))
    rgb = ls.shade(zi_clean, cmap=plt.get_cmap(CHOSEN_CMAP), vert_exag=0.1, blend_mode='soft')
    
    surf = ax.plot_surface(xi, yi, zi_clean, facecolors=rgb, linewidth=0, antialiased=True)
    
    ax.set_xlabel('Alpha (SSIM)', fontsize=12)
    ax.set_ylabel('Beta (MSE/MAE)', fontsize=12)
    ax.set_zlabel('Success Score', fontsize=12)
    ax.view_init(elev=25, azim=250)
    
    plt.title(f'3D Success Topology', fontsize=16)
    plt.savefig(target_dir / "success_score_3d.png", dpi=300)
    print(f"Ergebnisse in {target_dir} gespeichert.")
    plt.show()

if __name__ == "__main__":
    sel_folder = get_folder_selection()
    df_scores = extract_composite_scores(PARENT_CSV_DIR / sel_folder)
    
    if not df_scores.empty:
        plot_heatmaps(df_scores, sel_folder)
    else:
        print("Keine gültigen Daten gefunden.")