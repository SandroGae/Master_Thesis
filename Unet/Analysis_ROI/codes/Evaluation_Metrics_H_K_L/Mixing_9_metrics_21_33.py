import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & DATEN-BEREINIGUNG
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
CSV_FILE = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/CDW_Hyperparameter_Results_P0_P1.csv"
OUT_DIR  = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/Plots_P0_vs_P1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(CSV_FILE, comment='#', skip_blank_lines=True)
    
    # Umbenennung der Typen für die Legende und Anzeige
    df['Type'] = df['Type'].replace({
        'P0': 'Only MAE',
        'P1': 'Mixture MAE, MSE, SSIM'
    })
    
    # Filtern auf die relevanten Typen
    df = df[df['Type'].isin(['Only MAE', 'Mixture MAE, MSE, SSIM'])]
    
    for col in ['AreaRatio', 'SBRGain', 'PosShift', 'Seed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['SBRGain', 'AreaRatio'])
    
    # Die Formel für Detection Power
    df['DetectionPower'] = df['SBRGain'] * df['AreaRatio']
    df = df.sort_values(by=['Type', 'Seed'])

except Exception as e:
    print(f"Fehler: {e}")
    exit()

# =====================================================
# 2. STATISTIK PRINT-FUNKTION
# =====================================================
def print_model_statistics(data):
    print("\n" + "="*95)
    print(f"{'MODELL TYP':<25} | {'SEED':<6} | {'SBR GAIN':<12} | {'AREA RATIO':<12} | {'DET. POWER':<12}")
    print("-" * 95)
    for index, row in data.iterrows():
        print(f"{row['Type']:<25} | {int(row['Seed']):<6} | {row['SBRGain']:<12.4f} | {row['AreaRatio']:<12.4f} | {row['DetectionPower']:<12.4f}")
    
    print("-" * 95)
    summary = data.groupby('Type').agg({'SBRGain': 'mean', 'AreaRatio': 'mean', 'DetectionPower': 'mean'})
    for t in summary.index:
        print(f"TOTAL AVG {t:<15} | {'ALL':<6} | {summary.loc[t, 'SBRGain']:<12.4f} | {summary.loc[t, 'AreaRatio']:<12.4f} | {summary.loc[t, 'DetectionPower']:<12.4f}")
    print("="*95 + "\n")

# =====================================================
# 3. 2x3 BAR-CHART LAYOUT (UNTEREINANDER)
# =====================================================
def plot_comparison_matrix(data):
    # Layout: 2 Zeilen (MAE vs Mixture), 3 Spalten (DetPower, Sigma*A, Shift)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=120)
    
    types = ['Only MAE', 'Mixture MAE, MSE, SSIM']
    metrics = [
        ('DetectionPower', 'Detection Power ($SBR_{gain} \\cdot Area_{ratio}$)'),
        ('AreaRatio',      'Area Ratio ($\sigma \cdot A$)'),
        ('PosShift',       'Position Shift in $\mu$')
    ]
    
    # Farben für die Spalten
    colors = ['#3498db', '#2ecc71', '#e74c3c'] 

    # Globale Maxima für einheitliche Y-Achsen pro Spalte (bessere Vergleichbarkeit)
    limits = {
        'DetectionPower': data['DetectionPower'].max() * 1.15,
        'AreaRatio':      data['AreaRatio'].max() * 1.15,
        'PosShift':       data['PosShift'].max() * 1.15
    }

    for row_idx, t_name in enumerate(types):
        subset = data[data['Type'] == t_name]
        
        for col_idx, (col_name, title_part) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            x_labels = [f"S-{int(s)}" for s in subset['Seed']]
            x_pos = np.arange(len(x_labels))
            
            bars = ax.bar(x_pos, subset[col_name], color=colors[col_idx], 
                          edgecolor='black', alpha=0.8)
            
            # Formatierung
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_ylim(0, limits[col_name])
            
            # Titel nur in der ersten Zeile für die Metrik, 
            # in der zweiten Zeile zur Bestätigung des Typs
            full_title = f"{title_part}\n[{t_name}]"
            ax.set_title(full_title, fontsize=12, fontweight='bold')
            
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Werte über die Balken schreiben
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (limits[col_name] * 0.01),
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle("Performance Comparison: Only MAE vs. Mixture (MAE, MSE, SSIM)", 
                 fontsize=20, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    save_path = OUT_DIR / "MAE_vs_Mixture_Comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Vergleichs-Plot gespeichert unter: {save_path.name}")
    plt.show()

# =====================================================
# 4. START
# =====================================================
if __name__ == "__main__":
    print_model_statistics(df)
    plot_comparison_matrix(df)