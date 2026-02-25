from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Pfade & Setup
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR.parent / "Evaluation_Metrics_H_K_L" / "Full_Evaluation_Results.csv"
OUT_FILE = SCRIPT_DIR / "hyperparameter_heatmaps_refined.png"

# --------------------------------------------------
# 2. Daten laden & Aufbereiten
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

# Numerische Konvertierung
for col in ["Alpha", "Beta", "AreaRatio", "PosShift"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Alpha"] = df["Alpha"].round(4)
df["Beta"] = df["Beta"].round(4)
df["PosShiftAbs"] = df["PosShift"].abs()

# Einzigartige Achsenwerte
alpha_vals = np.sort(df["Alpha"].dropna().unique())
beta_vals  = np.sort(df["Beta"].dropna().unique())

# --- NEU: Daten-Kopie für Alpha = 1.0 ---
# Da bei Alpha=1 Beta keinen Einfluss hat, kopieren wir den Punkt (1,0) auf alle Beta-Werte bei Alpha=1
ref_a1_b0 = df[(df["Alpha"] == 1.0) & (df["Beta"] == 0.0)]
if not ref_a1_b0.empty:
    additionals = []
    for b in beta_vals:
        if b == 0.0: continue
        new_rows = ref_a1_b0.copy()
        new_rows["Beta"] = b
        additionals.append(new_rows)
    df = pd.concat([df] + additionals, ignore_index=True)

# --------------------------------------------------
# 3. Plotting Funktion (Visu-Match)
# --------------------------------------------------
def plot_refined_heatmaps(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Meshgrid für die glatte Interpolation (100x100)
    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xi, yi)

    # Konfiguration der 4 Subplots
    # (Metrik-Key, df_filter, colormap, Titel, Colorbar-Label)
    plot_configs = [
        ("AreaRatio", data, 'RdYlGn', "Area Ratio: All Runs", "Mean AreaRatio"),
        ("AreaRatio", data[data["IsClean"]==True], 'RdYlGn', "Area Ratio: Clean Seeds", "Mean AreaRatio"),
        ("PosShiftAbs", data, 'RdYlGn_r', "Peak Shift: All Runs", "Avg |Δμ|"),
        ("PosShiftAbs", data[data["IsClean"]==True], 'RdYlGn_r', "Peak Shift: Clean Seeds", "Avg |Δμ|")
    ]

    for i, (m_col, d_source, cmap, title, cb_label) in enumerate(plot_configs):
        ax = axes[i // 2, i % 2]
        
        # Aggregation auf Mittelwerte pro Hyperparameter-Punkt
        avg_data = d_source.groupby(['Alpha', 'Beta'])[m_col].mean().reset_index()
        
        # Interpolation (Cubic für den glatten Look)
        Z = griddata((avg_data['Alpha'], avg_data['Beta']), avg_data[m_col], (X, Y), method='cubic')
        
        # Hintergrund-Farbe
        cp = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        cb = fig.colorbar(cp, ax=ax)
        cb.set_label(cb_label, fontweight='bold')
        
        # Dezente Konturlinien einzeichnen
        ax.contour(X, Y, Z, levels=12, colors='black', alpha=0.1)
        
        # Die ursprünglichen 49 Messpunkte einzeichnen (Weiß mit schwarzem Rand)
        ax.scatter(avg_data['Alpha'], avg_data['Beta'], c='white', edgecolors='black', s=25, alpha=0.8, zorder=5)
        
        # Styling & Achsen
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(r"$\alpha$ (SSIM weight)", fontsize=11)
        ax.set_ylabel(r"$\beta$ (MSE vs MAE)", fontsize=11)
        
        # Achsen-Ticks exakt auf die 7x7 Werte setzen
        ax.set_xticks(alpha_vals)
        ax.set_yticks(beta_vals)
        ax.set_xticklabels([f"{v:.2f}" for v in alpha_vals], rotation=45)
        ax.set_yticklabels([f"{v:.2f}" for v in beta_vals])
        ax.set_aspect("equal")

    plt.suptitle("Hyperparameter Topology: Comparison of All vs. Clean Runs", 
                 fontsize=18, fontweight='bold', y=0.96)
    
    plt.savefig(OUT_FILE, bbox_inches='tight')
    plt.show()

# Start
plot_refined_heatmaps(df)
print(f"Heatmap erfolgreich erstellt: {OUT_FILE}")