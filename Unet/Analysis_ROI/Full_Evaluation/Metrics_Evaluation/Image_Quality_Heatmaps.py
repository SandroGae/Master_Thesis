import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Setup & Styling (Exakt wie dein XRD-Skript)
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Image_Quality_Metrics_Testset.csv" 
OUT_FILE = SCRIPT_DIR / "Quality_Heatmap_Comparison_7x7.png"

STYLE_PARAMS = {
    "SUBPLOT_TITLE_SIZE": 18,
    "TITLE_PAD": 15,
    "AXIS_LABEL_SIZE": 16,
    "TICK_LABEL_SIZE": 20,           # Große Achsen-Beschriftung
    "COLORBAR_LABEL_SIZE": 16,
    "COLORBAR_TICK_SIZE": 16,        # Große Colorbar-Ticks
    "COLORBAR_PAD": 0.08,
    "POINT_SIZE": 100,
    "SUBPLOT_WSPACE": 0.15,
}

# --------------------------------------------------
# 2. Datenverarbeitung & Punkt-Logik
# --------------------------------------------------
if not CSV_FILE.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {CSV_FILE}")

df_raw = pd.read_csv(CSV_FILE)

# Schritt A: Mittelwert über 10 Seeds pro Hyperparameter-Punkt (ergibt 43 Zeilen)
df_avg = df_raw.groupby(['Alpha', 'Beta', 'Point']).mean().reset_index()

# Schritt B: Lineare Skalierung pro Metrik (Bestwert unter den 43 Punkten = 1.0)
for m in ['clipped', 'unclipped']:
    ref_mae = df_avg[f'MAE_{m}'].min()
    ref_mse = df_avg[f'MSE_{m}'].min()
    ref_ssim = df_avg[f'SSIM_{m}'].max()
    
    # Lineares Verhältnis zum Punkt-Champion
    s_mae = ref_mae / df_avg[f'MAE_{m}']
    s_mse = ref_mse / df_avg[f'MSE_{m}']
    s_ssim = df_avg[f'SSIM_{m}'] / ref_ssim
    
    # Quality Score pro Punkt (Mittelwert der 3 linearen Scores)
    df_avg[f'QualityScore_{m}'] = (s_mae + s_mse + s_ssim) / 3.0

# Schritt C: Alpha=1.0 Fix für das 7x7 Grid (Kopiert Randpunkte)
unique_betas = np.sort(df_avg["Beta"].unique())
df_a1 = df_avg[df_avg["Alpha"] >= 0.99].copy()
df_rest = df_avg[df_avg["Alpha"] < 0.99]
replicated = []
for b in unique_betas:
    temp = df_a1.copy()
    temp["Beta"] = b
    replicated.append(temp)
df_final = pd.concat([df_rest] + replicated, ignore_index=True)

# --------------------------------------------------
# 3. Plotting (Side-by-Side)
# --------------------------------------------------
def plot_quality_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=200)
    
    xi = np.linspace(0, 1, 200); yi = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(xi, yi)
    fraction_labels = ['0', r'$\frac{1}{6}$', r'$\frac{2}{6}$', r'$\frac{3}{6}$', r'$\frac{4}{6}$', r'$\frac{5}{6}$', '1']
    alpha_vals = np.sort(df_avg["Alpha"].unique())
    beta_vals = np.sort(df_avg["Beta"].unique())

    configs = [
        ('QualityScore_unclipped', 'Quality Score: UNCLIPPED (MAE+MSE+SSIM)'),
        ('QualityScore_clipped', 'Quality Score: CLIPPED (MAE+MSE+SSIM)')
    ]

    for i, (col, title) in enumerate(configs):
        ax = axes[i]
        
        # Interpolation auf dem 7x7 Grid
        Z = griddata((data['Alpha'], data['Beta']), data[col], (X, Y), method='cubic')
        
        # Kontur-Plot (KEIN 'extend' -> keine Spitzen!)
        cp = ax.contourf(X, Y, Z, levels=15, cmap='RdYlGn', alpha=1.0)
        
        # Colorbar mit korrekten Font-Größen
        cbar = fig.colorbar(cp, ax=ax, format='%.2f', pad=STYLE_PARAMS["COLORBAR_PAD"])
        cbar.set_label('Score (Best Average = 1.0)', fontweight='bold', fontsize=STYLE_PARAMS["COLORBAR_LABEL_SIZE"])
        cbar.ax.tick_params(labelsize=STYLE_PARAMS["COLORBAR_TICK_SIZE"])
        
        # Alle 49 Messpunkte (7x7) einzeichnen
        ax.scatter(data['Alpha'], data['Beta'], c='white', edgecolors='black', 
                   s=STYLE_PARAMS["POINT_SIZE"], linewidths=1.5, zorder=5)

        # Labels & Titel Styling
        ax.set_title(r"$\mathbf{" + title + "}$", fontsize=STYLE_PARAMS["SUBPLOT_TITLE_SIZE"], pad=STYLE_PARAMS["TITLE_PAD"])
        ax.set_xlabel(r"$\mathbf{\alpha}$ $\mathbf{(SSIM)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
        ax.set_ylabel(r"$\mathbf{\beta}$ $\mathbf{(MSE/MAE)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
        
        ax.set_xticks(alpha_vals); ax.set_xticklabels(fraction_labels)
        ax.set_yticks(beta_vals); ax.set_yticklabels(fraction_labels)
        ax.tick_params(axis='both', which='major', labelsize=STYLE_PARAMS["TICK_LABEL_SIZE"])
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.subplots_adjust(wspace=STYLE_PARAMS["SUBPLOT_WSPACE"])
    plt.savefig(OUT_FILE, bbox_inches='tight')
    print(f"Heatmap erfolgreich unter {OUT_FILE.name} gespeichert.")

if __name__ == "__main__":
    plot_quality_comparison(df_final)