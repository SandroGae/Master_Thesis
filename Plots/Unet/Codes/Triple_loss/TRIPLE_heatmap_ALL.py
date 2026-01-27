import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# --- PFADE ---
PARENT_CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV")
FIG_BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Figures")

# --- EINSTELLUNGEN ---
CHOSEN_CMAP = 'plasma'
# Diese Ordner werden kombiniert
TARGET_FOLDERS = ["csv_triple_loss", "csv_triple_loss_deep_scan"]

def extract_combined_scores(base_path, folder_list):
    all_raw_data = []
    pattern = re.compile(r"a(\d+\.\d+)_b(\d+\.\d+)")
    
    for folder_name in folder_list:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            print(f"Hinweis: Ordner {folder_name} nicht gefunden. Überspringe...")
            continue
            
        print(f"Lese Daten aus: {folder_name}")
        for csv_file in folder_path.glob("log_*.csv"):
            match = pattern.search(csv_file.name)
            if match:
                try:
                    df = pd.read_csv(csv_file)
                    if 'val_loss' not in df.columns:
                        continue
                    
                    # Extraktion am Best-Point (Min Val Loss)
                    best_idx = df['val_loss'].idxmin()
                    best_row = df.loc[best_idx]
                    
                    all_raw_data.append({
                        'alpha': float(match.group(1)),
                        'beta': float(match.group(2)),
                        'mae': best_row['val_mae_center'],
                        'mse': best_row['val_mse_center'],
                        'ssim': best_row['val_ssim_center'],
                        'source': folder_name
                    })
                except Exception as e:
                    print(f"Fehler in {csv_file.name}: {e}")
    
    if not all_raw_data: return pd.DataFrame()
    
    df_combined = pd.DataFrame(all_raw_data)

    # Globale Normalisierung über alle Punkte aus beiden Ordnern
    def normalize(series, reverse=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min: return series * 0 + 1.0
        return (s_max - series) / (s_max - s_min) if reverse else (series - s_min) / (s_max - s_min)

    df_combined['n_mae'] = normalize(df_combined['mae'], reverse=True)
    df_combined['n_mse'] = normalize(df_combined['mse'], reverse=True)
    df_combined['n_ssim'] = normalize(df_combined['ssim'], reverse=False)

    # Success Score berechnen
    df_combined['score'] = (df_combined['n_mae'] + df_combined['n_mse'] + df_combined['n_ssim']) / 3.0
    return df_combined

def plot_combined_heatmaps(df, output_name):
    target_dir = FIG_BASE_DIR / output_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Grid für die Interpolation (Bereich 0-1 oder basierend auf Daten)
    # Nutze 0 bis 1, falls das dein definierter Parameterraum ist
    xi = np.linspace(0, 1, 200) 
    yi = np.linspace(0, 1, 200)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolation (Cubic für glatte Kurven)
    zi = griddata((df['alpha'], df['beta']), df['score'], (xi, yi), method='cubic')

    # --- 2D HEATMAP (FIXED: Added Contours) ---
    plt.figure(figsize=(12, 9))
    cp = plt.contourf(xi, yi, zi, levels=60, cmap=CHOSEN_CMAP)
    cbar = plt.colorbar(cp)
    cbar.set_label('Normalized Composite Score (NCS)', fontsize=14, labelpad=15)

    # Weiße Höhenlinien hinzufügen (wie im Original)
    contours = plt.contour(xi, yi, zi, levels=15, colors='white', alpha=0.3)
    plt.clabel(contours, inline=True, fontsize=9, fmt='%.2f')

    # Die Punkte einzeichnen
    plt.scatter(df['alpha'], df['beta'], c='white', s=40, edgecolors='black', alpha=0.8, label='Runs')
    
    plt.title(f'Combined Success Score Heatmap', fontsize=16, pad=20)
    plt.xlabel('Alpha (SSIM Weight)', fontsize=14)
    plt.ylabel('Beta (MSE/MAE Weight)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(target_dir / "combined_ncs_heatmap_2d.png", dpi=300)
    plt.close()

    # --- 3D TOPOLOGY (FIXED: Rotation & Style) ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(azdeg=315, altdeg=45)
    
    # zi gegen NaNs absichern
    zi_clean = np.nan_to_num(zi, nan=np.nanmin(zi))
    rgb = ls.shade(zi_clean, cmap=plt.get_cmap(CHOSEN_CMAP), vert_exag=0.1, blend_mode='soft')
    
    surf = ax.plot_surface(xi, yi, zi_clean, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
    
    # Achsen-Beschriftung analog zum Original
    ax.set_xlabel('Alpha (SSIM)', fontsize=12)
    ax.set_ylabel('Beta (MSE/MAE)', fontsize=12)
    ax.set_zlabel('Success Score', fontsize=12)
    
    # FIX: Rotation exakt wie im ersten Skript
    ax.view_init(elev=25, azim=250) 
    
    plt.title('3D Success Topology (Combined Data)', fontsize=16)
    plt.savefig(target_dir / "combined_ncs_topology_3d.png", dpi=300)
    
    print(f"Kombinierte Plots wurden unter {target_dir} gespeichert.")
    plt.show()
    
if __name__ == "__main__":
    df_all = extract_combined_scores(PARENT_CSV_DIR, TARGET_FOLDERS)
    
    if not df_all.empty:
        plot_combined_heatmaps(df_all, "triple_loss")
    else:
        print("Keine Daten zum Plotten gefunden.")