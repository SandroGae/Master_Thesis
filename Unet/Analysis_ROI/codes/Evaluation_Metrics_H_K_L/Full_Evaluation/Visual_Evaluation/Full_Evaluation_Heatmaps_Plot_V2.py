import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Setup & Datenvorbereitung
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv" 
OUT_FILE = SCRIPT_DIR / "Hyperparameter_Detailed_6Plot_Analysis.png"

# --- ZENTRALE STYLING PARAMETER ---
STYLE_PARAMS = {
    "SUBPLOT_TITLE_SIZE": 18,
    "TITLE_PAD": 15,
    "AXIS_LABEL_SIZE": 16,
    "TICK_LABEL_SIZE": 20,
    "COLORBAR_LABEL_SIZE": 16,
    "COLORBAR_TICK_SIZE": 16,
    "COLORBAR_PAD": 0.08,
    "LEGEND_TITLE_SIZE": 16,
    "LEGEND_TEXT_SIZE": 16,
    "POINT_SIZE": 100,
    "SUBPLOT_WSPACE": 0.15,
    "SUBPLOT_HSPACE": 0.25,
}

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# --- SCHRITT A: SPALTEN ERSTELLEN (Inkl. Ratios) ---
for d in ["H", "K", "L"]:
    df[f"Area_{d}"] = df[f"Amp_pr_{d}"] / df[f"Amp_gt_{d}"]
    df[f"Shift_{d}"] = (df[f"Mu_pr_{d}"] - df[f"Mu_gt_{d}"]).abs()
    # Ratios für den Report berechnen
    df[f"R2_Ratio_{d}"] = df[f"R2_pr_{d}"] / df[f"R2_gt_{d}"]
    df[f"RMSE_Ratio_{d}"] = df[f"RMSE_pr_{d}"] / df[f"RMSE_gt_{d}"]

# --- SCHRITT B: KONVERTIEREN ---
cols_to_fix = [
    "Alpha", "Beta", "Area_H", "Area_K", "Area_L", "Shift_H", "Shift_K", "Shift_L",
    "Sig_pr_H", "Sig_pr_K", "Sig_pr_L", "Sig_gt_H", "Sig_gt_K", "Sig_gt_L",
    "R2_Ratio_H", "R2_Ratio_K", "R2_Ratio_L", "RMSE_Ratio_H", "RMSE_Ratio_K", "RMSE_Ratio_L"
]
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Alpha"] = df["Alpha"].round(4)
df["Beta"] = df["Beta"].round(4)

alpha_vals = np.sort(df["Alpha"].dropna().unique())
beta_vals  = np.sort(df["Beta"].dropna().unique())

# --------------------------------------------------
# 2. Alpha=1.0 Fix
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

df = apply_alpha_fix(df)

# --------------------------------------------------
# 3. Aggregation & Ratio-basierte Filter-Definition
# --------------------------------------------------
# Diese Werte wurden durch den 2% cutoff gefunden
R2_RATIO_MIN = 0.5058
RMSE_RATIO_MAX = 1.1476

def get_stats(data_subset):
    # Logik für den Qualitäts-Filter (Sandra-Test)
    is_valid = (
        data_subset['Area_Val'].notna() & 
        (data_subset['R2_Ratio_Val'] > R2_RATIO_MIN) & 
        (data_subset['RMSE_Ratio_Val'] < RMSE_RATIO_MAX)
    )
    
    return pd.Series({
        'area_raw': data_subset['Area_Val'].mean(), # Alle Fits ohne Filter
        'area_penalized': data_subset['Area_Val'].where(is_valid, 0.0).mean(), # Mit Ratio-Filter
        'shift_filtered': data_subset['Shift_Val'].where(is_valid).mean(), # Verschiebung
        'fit_rate': is_valid.mean() * 100
    })

# --- DATEN VORBEREITEN ---
id_vars = ['Alpha', 'Beta', 'IsClean', 'Point', 'SeriesID']

melt_area = df.melt(id_vars=id_vars, value_vars=['Area_H', 'Area_K', 'Area_L'], var_name='Dir', value_name='Area_Val')
melt_r2 = df.melt(id_vars=id_vars, value_vars=['R2_pr_H', 'R2_pr_K', 'R2_pr_L'], var_name='Dir', value_name='R2_Val')
melt_rmse = df.melt(id_vars=id_vars, value_vars=['RMSE_pr_H', 'RMSE_pr_K', 'RMSE_pr_L'], var_name='Dir', value_name='RMSE_Val')
melt_amp_gt = df.melt(id_vars=id_vars, value_vars=['Amp_gt_H', 'Amp_gt_K', 'Amp_gt_L'], var_name='Dir', value_name='Amp_gt_Val')
melt_shift = df.melt(id_vars=id_vars, value_vars=['Shift_H', 'Shift_K', 'Shift_L'], var_name='Dir', value_name='Shift_Val')

# Ratios für den Report in den Analyse-DF einbetten
melt_r2_ratio = df.melt(id_vars=id_vars, value_vars=['R2_Ratio_H', 'R2_Ratio_K', 'R2_Ratio_L'], var_name='Dir', value_name='R2_Ratio_Val')
melt_rmse_ratio = df.melt(id_vars=id_vars, value_vars=['RMSE_Ratio_H', 'RMSE_Ratio_K', 'RMSE_Ratio_L'], var_name='Dir', value_name='RMSE_Ratio_Val')

df_analysis = melt_area.copy()
df_analysis['R2_Val'] = melt_r2['R2_Val']
df_analysis['RMSE_Val'] = melt_rmse['RMSE_Val']
df_analysis['Shift_Val'] = melt_shift['Shift_Val']
df_analysis['Amp_gt_Val'] = melt_amp_gt['Amp_gt_Val']
df_analysis['R2_Ratio_Val'] = melt_r2_ratio['R2_Ratio_Val']
df_analysis['RMSE_Ratio_Val'] = melt_rmse_ratio['RMSE_Ratio_Val']

# --- STATISTIKEN BERECHNEN ---
grid_stats = df_analysis.groupby(['Alpha', 'Beta']).apply(get_stats).reset_index()
grid_stats_clean = df_analysis[df_analysis['IsClean'] == True].groupby(['Alpha', 'Beta']).apply(get_stats).reset_index()
stability = df.groupby(['Alpha', 'Beta'])['IsClean'].mean().reset_index(name='stability_rate')
stability['stability_rate'] *= 100

stats = pd.merge(grid_stats, grid_stats_clean, on=['Alpha', 'Beta'], suffixes=('_all', '_clean'))
stats = pd.merge(stats, stability, on=['Alpha', 'Beta'])

stats = stats.rename(columns={'shift_filtered_all': 'shift_all', 'shift_filtered_clean': 'shift_clean'})

# --------------------------------------------------
# 4. Plotting (EXAKT WIE GEWÜNSCHT)
# --------------------------------------------------
def plot_final_6_heatmaps(data):
    fig, axes = plt.subplots(3, 2, figsize=(20, 26), dpi=200)

    xi = np.linspace(0, 1, 100); yi = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xi, yi)
    fraction_labels = ['0', r'$\frac{1}{6}$', r'$\frac{2}{6}$', r'$\frac{3}{6}$', r'$\frac{4}{6}$', r'$\frac{5}{6}$', '1']

    plot_configs = [
        # ZEILE 1: ROHDATEN (Alle Fits)
        ('area_raw_all', 'RdYlGn', 'Area: All Series (Raw)', '(GT=1.0)', False),
        ('area_raw_clean', 'RdYlGn', 'Area: Converged Models (Raw)', '(GT=1.0)', True),
        
        # ZEILE 2: QUALITÄTS-FILTER (R2/RMSE Ratios)
        ('area_penalized_all', 'RdYlGn', 'Area: All Series (Ratio Filter)', '(GT=1.0)', False),
        ('area_penalized_clean', 'RdYlGn', 'Area: Converged (Ratio Filter)', '(GT=1.0)', True),
        
        # ZEILE 3: PEAK SHIFT
        ('shift_all', 'RdYlGn_r', 'Peak Shift: All Series', 'Pixels', False),
        ('shift_clean', 'RdYlGn_r', 'Peak Shift: Converged Models', 'Pixels', True)
    ]

    for i, (m_col, cmap, title, cb_label, use_halo) in enumerate(plot_configs):
        ax = axes[i // 2, i % 2]
        valid = data.dropna(subset=[m_col])
        if valid.empty: continue
        
        Z = griddata((valid['Alpha'], valid['Beta']), valid[m_col], (X, Y), method='cubic')
        cp = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=1.0)
        
        cb_format = '%.2f' if cb_label == 'Pixels' else '%.3f'
        cbar = fig.colorbar(cp, ax=ax, format=cb_format, pad=STYLE_PARAMS["COLORBAR_PAD"])
        cbar.set_label(cb_label, fontweight='bold', fontsize=STYLE_PARAMS["COLORBAR_LABEL_SIZE"])
        cbar.ax.tick_params(labelsize=STYLE_PARAMS["COLORBAR_TICK_SIZE"])

        for _, row in data.iterrows():
            if i == 2:
                ax.text(row['Alpha'] + 0.02, row['Beta'] + 0.01, f"{int(row['fit_rate_all'])}%", 
                        color='black', fontsize=12, ha='left', va='center', fontweight='bold', zorder=10)

            if use_halo and row['stability_rate'] < 100:
                halo_size = (100 - row['stability_rate']) * 110 
                dynamic_alpha = (1.0 - row['stability_rate']/100.0) * 0.6
                ax.scatter(row['Alpha'], row['Beta'], s=halo_size, color='blue', 
                            alpha=dynamic_alpha, edgecolors='none', zorder=3)

            p_color = 'white' if (row['stability_rate'] == 100) else '#dcdcdc'
            ax.scatter(row['Alpha'], row['Beta'], c=p_color, edgecolors='black', 
                        s=STYLE_PARAMS["POINT_SIZE"], linewidths=1.5, zorder=5)

        ax.set_title(title, fontsize=STYLE_PARAMS["SUBPLOT_TITLE_SIZE"], pad=STYLE_PARAMS["TITLE_PAD"], fontweight='bold')
        ax.set_xlabel(r"$\mathbf{\alpha}$ $\mathbf{(SSIM)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
        ax.set_ylabel(r"$\mathbf{\beta}$ $\mathbf{(MSE/MAE)}$", fontsize=STYLE_PARAMS["AXIS_LABEL_SIZE"], fontweight='bold')
        
        ax.set_xticks(alpha_vals); ax.set_xticklabels(fraction_labels)
        ax.set_yticks(beta_vals); ax.set_yticklabels(fraction_labels)
        ax.tick_params(axis='both', which='major', labelsize=STYLE_PARAMS["TICK_LABEL_SIZE"])
        ax.set_aspect("equal")

    leg_el = [Line2D([0], [0], marker='o', color='w', label='100%', 
                      markerfacecolor='white', markeredgecolor='black', markersize=12)]
    min_stable = int(np.floor(data['stability_rate'].min() / 10.0) * 10)
    for s in range(90, min_stable - 1, -10):
        a = (1.0 - s/100.0) * 0.65
        leg_el.append(Line2D([0], [0], marker='o', color='none', label=f'{s}%',
                              markerfacecolor='blue', alpha=a, markersize=15))

    fig.legend(handles=leg_el, loc='upper center', bbox_to_anchor=(0.5, 0.985), 
                ncol=len(leg_el), fontsize=STYLE_PARAMS["LEGEND_TEXT_SIZE"], 
                title="Model Stability Indicator", 
                title_fontsize=STYLE_PARAMS["LEGEND_TITLE_SIZE"], frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.subplots_adjust(wspace=STYLE_PARAMS["SUBPLOT_WSPACE"], hspace=STYLE_PARAMS["SUBPLOT_HSPACE"])
    plt.savefig(OUT_FILE, bbox_inches='tight')
    print(f"Erfolgreich gespeichert: {OUT_FILE}")

plot_final_6_heatmaps(stats)

# --------------------------------------------------
# 5. Global Failure Report (KORRIGIERT)
# --------------------------------------------------
def print_global_failure_report(df_an, r2_ratio_limit, rmse_ratio_limit):
    # Fehler-Flags basierend auf Ratios
    df_an['Fail_R2']   = (df_an['R2_Ratio_Val'] < r2_ratio_limit) & df_an['R2_Ratio_Val'].notna()
    df_an['Fail_RMSE'] = (df_an['RMSE_Ratio_Val'] > rmse_ratio_limit) & df_an['RMSE_Ratio_Val'].notna()
    df_an['Fail_NaN']  = df_an['R2_Val'].isna()
    df_an['Any_Fail']  = df_an['Fail_R2'] | df_an['Fail_RMSE'] | df_an['Fail_NaN']

    report = df_an.groupby('SeriesID').agg({
        'Any_Fail': 'sum',
        'Fail_R2': 'sum',
        'Fail_RMSE': 'sum',
        'Fail_NaN': 'sum',
        'R2_Ratio_Val': 'mean',
        'RMSE_Ratio_Val': 'mean',
        'R2_Val': 'count'
    }).rename(columns={'R2_Val': 'Total_Attempts'}).reset_index()

    print("\n" + "="*130)
    print(f"{'GLOBALER XRD-FIT FEHLERBERICHT & RATIOS':^130}")
    # FIX: Hier r2_ratio_limit statt r2_limit verwenden
    print(f"{f'(Filter: R²-Ratio < {r2_ratio_limit} | RMSE-Ratio > {rmse_ratio_limit})':^130}")
    print("="*130)
    
    header = f"{'Serie':<8} | {'Gesamt-Fails':<12} | {'R²-Ratio-F':<10} | {'RMSE-Ratio-F':<12} | {'NaN-Fail':<8} | {'R²-Ratio-M':<10} | {'RMSE-Ratio-M':<10} | {'Erfolg':<8}"
    print(header)
    print("-" * len(header))

    for _, row in report.sort_values('Any_Fail', ascending=False).iterrows():
        success_rate = (1 - (row['Any_Fail'] / row['Total_Attempts'])) * 100
        print(f"S-{int(row['SeriesID']):<3} | "
              f"{int(row['Any_Fail']):<12} | "
              f"{int(row['Fail_R2']):<10} | "
              f"{int(row['Fail_RMSE']):<12} | "
              f"{int(row['Fail_NaN']):<8} | "
              f"{row['R2_Ratio_Val']:<10.3f} | "
              f"{row['RMSE_Ratio_Val']:<10.3f} | "
              f"{success_rate:>7.1f}%")

    print("-" * len(header))
    total_fails = report['Any_Fail'].sum()
    total_att = report['Total_Attempts'].sum()
    print(f"{'TOTAL':<8} | "
          f"{int(total_fails):<12} | "
          f"{int(report['Fail_R2'].sum()):<8} | "
          f"{int(report['Fail_RMSE'].sum()):<10} | "
          f"{int(report['Fail_NaN'].sum()):<8} | "
          f"{report['R2_Ratio_Val'].mean():<10.3f} | "
          f"{report['RMSE_Ratio_Val'].mean():<10.3f} | "
          f"{(1 - (total_fails/total_att))*100:>7.1f}%")
    print("="*130)

print_global_failure_report(df_analysis, R2_RATIO_MIN, RMSE_RATIO_MAX)