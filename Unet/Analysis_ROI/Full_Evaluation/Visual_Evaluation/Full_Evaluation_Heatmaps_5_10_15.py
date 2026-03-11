import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Setup & Datenvorbereitung
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv"
OUT_FILE = SCRIPT_DIR / "Convergence_Analysis.png"

# --- ZENTRALE STYLING PARAMETER (IDENTISCH ZUM HAUPTSKRIPT) ---
STYLE_PARAMS = {
    "SUBPLOT_TITLE_SIZE": 18,
    "TITLE_PAD": 15,
    "AXIS_LABEL_SIZE": 16,
    "TICK_LABEL_SIZE": 20,
    "COLORBAR_LABEL_SIZE": 16,
    "COLORBAR_TICK_SIZE": 16,
    "COLORBAR_PAD": 0.08,
    "POINT_SIZE": 100,  # Erhöht für Konsistenz
    "SUBPLOT_WSPACE": 0.15,
    "SUBPLOT_HSPACE": 0.25,
}

RANDOM_SEED = 43
random.seed(RANDOM_SEED)
SUBSET_SIZES = [5, 10, 15, 20, 25, 30]

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {CSV_FILE}")

# --- 1. Daten laden ---
df_master = pd.read_csv(CSV_FILE)

# --- 2. Zuerst alles in Zahlen umwandeln (Schritt B vorziehen) ---
cols_to_fix = ["Alpha", "Beta", "Amp_pr_H", "Sig_pr_H", "Amp_gt_H", "Sig_gt_H",
               "Amp_pr_K", "Sig_pr_K", "Amp_gt_K", "Sig_gt_K",
               "Amp_pr_L", "Sig_pr_L", "Amp_gt_L", "Sig_gt_L"]

for col in cols_to_fix:
    if col in df_master.columns:
        df_master[col] = pd.to_numeric(df_master[col], errors="coerce")

# --- 3. Dann die Fläche berechnen (Schritt A) ---
for d in ["H", "K", "L"]:
    # Jetzt sind Amp und Sig garantiert Zahlen -> Multiplikation sicher
    df_master[f"Area_{d}"] = (df_master[f"Amp_pr_{d}"] * df_master[f"Sig_pr_{d}"]) / \
                             (df_master[f"Amp_gt_{d}"] * df_master[f"Sig_gt_{d}"])

df_master["Alpha"] = df_master["Alpha"].round(4)
df_master["Beta"] = df_master["Beta"].round(4)
alpha_vals = np.sort(df_master["Alpha"].dropna().unique())
beta_vals  = np.sort(df_master["Beta"].dropna().unique())
ALL_SERIES_IDS = sorted(df_master['SeriesID'].unique())

# --------------------------------------------------
# 2. Hilfsfunktionen
# --------------------------------------------------
def apply_alpha_fix(dataframe):
    df_a1 = dataframe[dataframe["Alpha"] == 1.0].copy()
    if df_a1.empty: return dataframe
    dataframe = dataframe[dataframe["Alpha"] != 1.0]
    additionals = []
    for b in beta_vals:
        new_rows = df_a1.copy()
        new_rows["Beta"] = b
        additionals.append(new_rows)
    return pd.concat([dataframe] + additionals, ignore_index=True)

def get_penalized_stats(data_subset):
    df_melted = data_subset.melt(id_vars=['Alpha', 'Beta'], 
                                 value_vars=['Area_H', 'Area_K', 'Area_L'], 
                                 value_name='Area_Val')
    # Penalized: NaN = 0.0
    stats = df_melted.groupby(['Alpha', 'Beta'])['Area_Val'].apply(
        lambda x: x.fillna(0.0).mean()
    ).reset_index(name='area_penalized')
    return apply_alpha_fix(stats)

# --------------------------------------------------
# 3. Globale Extremwerte ermitteln (Pre-Processing)
# --------------------------------------------------
print(">>> Analysiere Daten für maximalen Kontrast...")
all_subset_results = []
for size in SUBSET_SIZES:
    current_ids = random.sample(ALL_SERIES_IDS, size)
    df_sub = df_master[df_master['SeriesID'].isin(current_ids)].copy()
    all_subset_results.append(get_penalized_stats(df_sub))

global_min = min(res['area_penalized'].min() for res in all_subset_results)
global_max = max(res['area_penalized'].max() for res in all_subset_results)

# --------------------------------------------------
# 4. Haupt-Plotting
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(28, 16), dpi=200) # figsize angepasst
xi = np.linspace(0, 1, 150); yi = np.linspace(0, 1, 150)
X, Y = np.meshgrid(xi, yi)
fraction_labels = ['0', r'$\frac{1}{6}$', r'$\frac{2}{6}$', r'$\frac{3}{6}$', r'$\frac{4}{6}$', r'$\frac{5}{6}$', '1']

for idx, size in enumerate(SUBSET_SIZES):
    ax = axes[idx // 3, idx % 3]
    plot_data = all_subset_results[idx]
    
    Z = griddata((plot_data['Alpha'], plot_data['Beta']), 
                 plot_data['area_penalized'], (X, Y), method='cubic')
    
    cp = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn', vmin=global_min, vmax=global_max)
    
    cbar = fig.colorbar(cp, ax=ax, format='%.2f', pad=STYLE_PARAMS["COLORBAR_PAD"])
    cbar.set_label('(GT=1.0)', fontweight='bold', fontsize=STYLE_PARAMS["COLORBAR_LABEL_SIZE"])
    cbar.ax.tick_params(labelsize=STYLE_PARAMS["COLORBAR_TICK_SIZE"])

    # Scatter-Points konsistent zum Hauptskript (weiß mit schwarzem Rand)
    ax.scatter(plot_data['Alpha'], plot_data['Beta'], c='white', 
               edgecolors='black', s=STYLE_PARAMS["POINT_SIZE"], linewidths=1.5, zorder=5)

    # Titel ohne LaTeX für korrekte Abstände
    ax.set_title(f"Average Area Over {size} Randomly Picked Series", 
                 fontsize=STYLE_PARAMS["SUBPLOT_TITLE_SIZE"], fontweight='bold', pad=STYLE_PARAMS["TITLE_PAD"])
    
    # Achsenbeschriftung konsistent mit LaTeX-Formeln
    ax.set_xlabel(r"$\mathbf{\alpha}$ $\mathbf{(SSIM)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
    ax.set_ylabel(r"$\mathbf{\beta}$ $\mathbf{(MSE/MAE)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
    
    ax.set_xticks(alpha_vals); ax.set_xticklabels(fraction_labels)
    ax.set_yticks(beta_vals); ax.set_yticklabels(fraction_labels)
    ax.tick_params(axis='both', which='major', labelsize=STYLE_PARAMS["TICK_LABEL_SIZE"])
    ax.set_aspect("equal")

# suptitle entfernt wie gewünscht
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.subplots_adjust(wspace=STYLE_PARAMS["SUBPLOT_WSPACE"], hspace=STYLE_PARAMS["SUBPLOT_HSPACE"])

plt.savefig(OUT_FILE, bbox_inches='tight')
print(f"Erfolgreich gespeichert: {OUT_FILE}")