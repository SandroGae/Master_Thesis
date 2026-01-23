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
# Wähle hier deine Wunsch-Farbe: 'plasma', 'magma', 'inferno', 'coolwarm', 'RdYlBu_r'
CHOSEN_CMAP = 'plasma'

def get_folder_selection():
    subdirs = sorted([d.name for d in PARENT_CSV_DIR.iterdir() if d.is_dir()])
    print("\nWelchen Ordner für die Heatmap verwenden?")
    for i, d in enumerate(subdirs):
        print(f"[{i}] {d}")
    idx = int(input("\nAuswahl: "))
    return subdirs[idx]

def extract_values(folder_path):
    results = []
    pattern = re.compile(r"a(\d+\.\d+)_b(\d+\.\d+)")
    
    for csv_file in folder_path.glob("*.csv"):
        if csv_file.name.startswith("summary"): continue
        match = pattern.search(csv_file.name)
        if match:
            alpha = float(match.group(1))
            beta = float(match.group(2))
            df = pd.read_csv(csv_file)
            if 'val_psnr_center' in df.columns:
                best_psnr = df['val_psnr_center'].max()
                results.append({'alpha': alpha, 'beta': beta, 'psnr': best_psnr})
    return pd.DataFrame(results)

def plot_heatmaps(df, folder_name):
    target_dir = FIG_BASE_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Grid für höhere Auflösung (Interpolation)
    xi = np.linspace(0, 1, 200) # Erhöht auf 200 für glattere Optik
    yi = np.linspace(0, 1, 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((df['alpha'], df['beta']), df['psnr'], (xi, yi), method='cubic')

    # --- PLOT 1: 2D Heatmap (Verbessert) ---
    fig_2d = plt.figure(figsize=(12, 9))
    
    # Heatmap mit neuer Farbe und mehr Stufen
    cp = plt.contourf(xi, yi, zi, levels=60, cmap=CHOSEN_CMAP)
    cbar = plt.colorbar(cp)
    cbar.set_label('Best PSNR (dB)', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Messpunkte (etwas dezenter)
    plt.scatter(df['alpha'], df['beta'], c='white', s=30, edgecolors='black', linewidth=0.8, alpha=0.7, label='Runs')
    
    # Feine Konturlinien für Topografie-Look
    contours = plt.contour(xi, yi, zi, levels=15, colors='white', alpha=0.3, linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=9, fmt='%.1f')

    plt.title(f'Triple Loss Performance Landscape\nFolder: {folder_name}', fontsize=16, pad=20)
    plt.xlabel('Alpha (SSIM Weight)', fontsize=14)
    plt.ylabel('Beta (MSE vs MAE Weight)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    save_path_2d = target_dir / "heatmap_2d_plasma.png"
    plt.savefig(save_path_2d, dpi=300)
    print(f"2D Plot gespeichert: {save_path_2d}")
    plt.close(fig_2d)

    # --- PLOT 2: 3D Surface Plot (Massiv Verbessert) ---
    fig_3d = plt.figure(figsize=(14, 10))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    # Beleuchtung für plastischen Effekt
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(zi, cmap=plt.get_cmap(CHOSEN_CMAP), vert_exag=0.1, blend_mode='soft')
    
    # Surface Plot mit Shading
    surf = ax.plot_surface(xi, yi, zi, facecolors=rgb, linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=1.0)
    
    # Achsenbeschriftung (Gross und klar)
    ax.set_xlabel('Alpha (SSIM Weight)', fontsize=14, labelpad=15)
    ax.set_ylabel('Beta (MSE/MAE Weight)', fontsize=14, labelpad=15)
    ax.set_zlabel('PSNR (dB)', fontsize=14, labelpad=15, rotation=90)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='z', which='major', labelsize=11, pad=10)

    ax.set_title(f'3D Performance Topology: {folder_name}', fontsize=16, pad=20)
    
    # Optimaler Blickwinkel für die "Landschaft"
    ax.view_init(elev=35, azim=230)
    
    # Colorbar
    m = plt.cm.ScalarMappable(cmap=CHOSEN_CMAP)
    m.set_array(zi)
    cbar = plt.colorbar(m, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label('PSNR (dB)', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    save_path_3d = target_dir / "surface_3d_plasma.png"
    plt.savefig(save_path_3d, dpi=300)
    print(f"3D Plot gespeichert: {save_path_3d}")
    plt.show()

if __name__ == "__main__":
    folder_name = get_folder_selection()
    csv_folder_path = PARENT_CSV_DIR / folder_name
    
    data_df = extract_values(csv_folder_path)
    if not data_df.empty:
        plot_heatmaps(data_df, folder_name)
    else:
        print("Keine Daten gefunden.")