from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Setup & Pfade
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR.parent / "Evaluation_Metrics_H_K_L" / "Full_Evaluation_Results.csv"
OUT_FILE = SCRIPT_DIR / "Hyperparameter_Heatmaps_3D_Complete.png"

df = pd.read_csv(CSV_FILE)

# Reinigung & Konvertierung
cols_to_fix = ["Alpha", "Beta", "Area_H", "Area_K", "Area_L", "Shift_H", "Shift_K", "Shift_L"]
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Mittelwert über H, K, L berechnen
df['Area_Avg'] = df[['Area_H', 'Area_K', 'Area_L']].mean(axis=1)
df['Shift_Avg'] = df[['Shift_H', 'Shift_K', 'Shift_L']].mean(axis=1)

df["Alpha"] = df["Alpha"].round(4)
df["Beta"] = df["Beta"].round(4)

# Achsenwerte ermitteln
alpha_vals = np.sort(df["Alpha"].dropna().unique())
beta_vals  = np.sort(df["Beta"].dropna().unique())

# --------------------------------------------------
# 2. Alpha=1.0 Fix: Kopieren der Seeds für das 7x7 Gitter
# --------------------------------------------------
# Wir nehmen die Daten von Alpha=1.0 (da Beta hier egal ist) und
# duplizieren sie für alle Beta-Stufen, um das Gitter rechts zu schließen.
df_a1 = df[df["Alpha"] == 1.0].copy()
if not df_a1.empty:
    # Wir löschen die Original-Alpha=1 Einträge kurzzeitig aus dem Haupt-DF,
    # um sie dann sauber für alle Beta-Werte neu einzufügen
    df = df[df["Alpha"] != 1.0]
    
    additionals = []
    for b in beta_vals:
        new_rows = df_a1.copy()
        new_rows["Beta"] = b
        additionals.append(new_rows)
    df = pd.concat([df] + additionals, ignore_index=True)

# --------------------------------------------------
# 3. Aggregation
# --------------------------------------------------
grid_stats = df.groupby(['Alpha', 'Beta']).agg(
    total_seeds=('IsClean', 'count'),
    clean_seeds=('IsClean', lambda x: x.sum()),
    area_all=('Area_Avg', 'mean'),
    area_clean=('Area_Avg', lambda x: x[df.loc[x.index, 'IsClean'] == True].mean()),
    shift_all=('Shift_Avg', 'mean'),
    shift_clean=('Shift_Avg', lambda x: x[df.loc[x.index, 'IsClean'] == True].mean())
).reset_index()

grid_stats['SuccessRate'] = (grid_stats['clean_seeds'] / grid_stats['total_seeds']) * 100
grid_stats['SuccessRate'] = grid_stats['SuccessRate'].round(6)

# --------------------------------------------------
# 4. Plotting
# --------------------------------------------------
def plot_final_heatmaps(stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 15), dpi=250)
    plt.subplots_adjust(wspace=0.2, hspace=0.15)

    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xi, yi)

    plot_configs = [
        ('area_all', 'RdYlGn', r"$\mathbf{Area Ratio}$: Global (All Seeds)", "Mean AreaRatio (3D Avg)", False),
        ('area_clean', 'RdYlGn', r"$\mathbf{Area Ratio}$: Clean Seeds (Success Aura)", "Mean AreaRatio (3D Avg)", True),
        ('shift_all', 'RdYlGn_r', r"$\mathbf{Peak Shift}$: Global (All Seeds)", "Avg |Δμ| (3D Avg)", False),
        ('shift_clean', 'RdYlGn_r', r"$\mathbf{Peak Shift}$: Clean Seeds (Success Aura)", "Avg |Δμ| (3D Avg)", True)
    ]

    halo_cmap = plt.cm.Blues
    halo_success_values = np.sort(stats.loc[stats['SuccessRate'] < 100, 'SuccessRate'].unique())

    def halo_color_for_success(success_value):
        if success_value >= 100: return None
        if len(halo_success_values) == 0: return halo_cmap(0.5)
        min_success = halo_success_values.min()
        t = 0.0 if np.isclose(min_success, 90.0) else (90.0 - success_value) / (90.0 - min_success)
        return halo_cmap(0.25 + 0.70 * np.clip(t, 0, 1))

    for i, (m_col, cmap, title, cb_label, show_aura) in enumerate(plot_configs):
        ax = axes[i // 2, i % 2]
        
        valid_mask = ~stats[m_col].isna()
        Z = griddata((stats['Alpha'][valid_mask], stats['Beta'][valid_mask]), 
                     stats[m_col][valid_mask], (X, Y), method='cubic')

        cp = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.9)
        cb = fig.colorbar(cp, ax=ax)
        cb.set_label(cb_label, fontweight='bold')
        ax.contour(X, Y, Z, levels=12, colors='black', alpha=0.1)

        for _, row in stats.iterrows():
            sr = row['SuccessRate']
            if show_aura and sr < 100:
                color = halo_color_for_success(sr)
                halo_size = ((100 - sr) / 100 * 4000)
                ax.scatter(row['Alpha'], row['Beta'], s=halo_size, color=color,
                           alpha=0.4, edgecolors=color, linewidths=1, zorder=3)
            
            p_color = 'white' if sr == 100 else '#f0f0f0'
            ax.scatter(row['Alpha'], row['Beta'], c=p_color, edgecolors='black', s=35, zorder=5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(r"$\alpha$ (SSIM weight)", fontsize=11)
        ax.set_ylabel(r"$\beta$ (MSE vs MAE)", fontsize=11)
        ax.set_xticks(alpha_vals)
        ax.set_yticks(beta_vals)
        ax.set_aspect("equal")

    # Legende
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='100%',
                       markerfacecolor='white', markeredgecolor='black', markersize=8)]
    for success in sorted(halo_success_values, reverse=True):
        color = halo_color_for_success(success)
        legend_elements.append(mpatches.Patch(color=color, alpha=0.5, label=f"{int(success)}%"))
    
    leg = fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.58),
                     title="Model Stability", fontsize=10)
    plt.setp(leg.get_title(), fontweight='bold')    
    
    plt.suptitle("Complete 7x7 Hyperparameter Topology (3D Averaged: H, K, L)\n"
                 "Duplicated Alpha=1.0 results across all Beta values for grid consistency", 
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(OUT_FILE, bbox_inches='tight')
    print("Speichern erfolgreich:", OUT_FILE)
    plt.show()

plot_final_heatmaps(grid_stats)