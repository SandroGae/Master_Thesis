import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LightSource
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
CSV_FILE = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/CDW_Hyperparameter_Results.csv"
OUT_DIR  = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/Hyperparameter_Topologies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Laden der Daten
if not CSV_FILE.exists():
    raise FileNotFoundError(f"Die Datei {CSV_FILE} wurde nicht gefunden. Bitte erst das Evaluations-Skript ausführen.")

df = pd.read_csv(CSV_FILE)

# Definition der Metriken (Identisch zum gewünschten visuellen Stil)
metrics = {
    'AreaRatio': ('Area Ratio ($A \\cdot \\sigma$)', 'RdYlGn'), 
    'SBRGain':   ('Denoising Gain', 'plasma'), 
    'PosShift':  ('Positional Shift [px]', 'magma')
}

directions = ["H", "K", "L"]

# =====================================================
# 2. PLOTTING FUNKTION (ANALOG ZUM REFERENZ-CODE)
# =====================================================
def generate_topologies():
    print(f"Starte Visualisierung aus {CSV_FILE.name}...")
    
    for m_key, (m_title, cmap_name) in metrics.items():
        for d in directions:
            # Daten für Richtung filtern
            sub = df[df['Dir'] == d]
            if sub.empty: continue

            # --- INTERPOLATION (für den glatten Look) ---
            # Wir erstellen ein feines Gitter von 0 bis 1
            xi = np.linspace(0, 1, 100)
            yi = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(xi, yi)
            
            # Interpolation der unregelmäßigen Alpha/Beta Punkte auf das Gitter
            Z = griddata((sub['Alpha'], sub['Beta']), sub[m_key], (X, Y), method='cubic')
            
            # Nan-Werte bereinigen (für die 3D Darstellung)
            Z_clean = np.nan_to_num(Z, nan=np.nanmin(Z))

            # --- A) 2D SMOOTH HEATMAP (Contourf) ---
            plt.figure(figsize=(10, 8), dpi=150)
            # 50 Level für extrem glatte Übergänge
            cp = plt.contourf(X, Y, Z, levels=50, cmap=cmap_name)
            plt.colorbar(cp, label=m_title)
            
            # Dezente Konturlinien hinzufügen
            plt.contour(X, Y, Z, levels=15, colors='white', alpha=0.2)
            
            # Die tatsächlichen Datenpunkte als Scatter einzeichnen
            plt.scatter(sub['Alpha'], sub['Beta'], c='white', edgecolors='black', s=30, 
                        label='Model Data Points', zorder=5)
            
            plt.title(f"{m_title} - Direction {d}\n(Topographical 2D Map)", fontsize=14, fontweight='bold')
            plt.xlabel("Alpha (Total Loss Weight / SSIM)", fontsize=12)
            plt.ylabel("Beta (Structural Weight / MSE)", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.3)
            
            plt.savefig(OUT_DIR / f"Heatmap_2D_{m_key}_{d}.png", bbox_inches='tight')
            plt.close()

            # --- B) 3D TOPOLOGY (Surface mit LightSource) ---
            fig = plt.figure(figsize=(12, 9), dpi=150)
            ax = fig.add_subplot(111, projection='3d')
            
            # Lichtquelle definieren (für die plastische Darstellung)
            ls = LightSource(azdeg=315, altdeg=45)
            
            # Shading berechnen
            rgb = ls.shade(Z_clean, cmap=plt.get_cmap(cmap_name), vert_exag=0.1, blend_mode='soft')
            
            # Oberfläche plotten
            surf = ax.plot_surface(X, Y, Z_clean, facecolors=rgb, linewidth=0, 
                                   antialiased=True, shade=False)
            
            # Blickwinkel einstellen (analog zur Referenz)
            ax.view_init(elev=25, azim=250)
            
            ax.set_title(f"3D Topology: {m_title} - Direction {d}", fontsize=15, fontweight='bold')
            ax.set_xlabel('Alpha (SSIM)')
            ax.set_ylabel('Beta (MSE)')
            ax.set_zlabel(m_key)
            
            plt.savefig(OUT_DIR / f"Topology_3D_{m_key}_{d}.png", bbox_inches='tight')
            plt.close()
            
            print(f" OK: Metrik {m_key} | Richtung {d}")

if __name__ == "__main__":
    generate_topologies()
    print(f"\nFertig! Alle Plots wurden in {OUT_DIR} gespeichert.")