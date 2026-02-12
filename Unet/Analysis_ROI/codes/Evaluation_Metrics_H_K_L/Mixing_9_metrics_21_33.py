import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    df = df[df['Type'] != 'Type'] # Header-Dubletten entfernen
    
    # Konvertierung in Zahlen
    for col in ['AreaRatio', 'SBRGain', 'PosShift', 'Seed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['SBRGain', 'AreaRatio'])
    df['DetectionPower'] = df['SBRGain'] * df['AreaRatio']
    
    # WICHTIG: Nach Seed sortieren (niedrigster links)
    df = df.sort_values(by='Seed')
    
except Exception as e:
    print(f"Fehler: {e}")
    exit()

# =====================================================
# 2. 2x2 BAR-CHART LAYOUT
# =====================================================
def plot_2x2_bars(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    
    # Daten nach Typ trennen
    p0_data = data[data['Type'] == 'P0']
    p1_data = data[data['Type'] == 'P1']
    
    # Metriken und Achsen-Zuordnung
    # Format: (Datenquelle, Metrik-Name, Titel, Farbe, Plot-Position)
    plot_map = [
        (p0_data, 'DetectionPower', 'Detection Power: P0 (Baseline)', '#3498db', (0, 0)),
        (p1_data, 'DetectionPower', 'Detection Power: P1 (Optimized)', '#e67e22', (0, 1)),
        (p0_data, 'PosShift',       'Position Shift: P0 (Baseline)',  '#e74c3c', (1, 0)),
        (p1_data, 'PosShift',       'Position Shift: P1 (Optimized)', '#c0392b', (1, 1))
    ]

    # Globale Maxima für einheitliche Achsen (Y-Achse vergleichbar machen)
    max_dp = data['DetectionPower'].max() * 1.1
    max_ps = data['PosShift'].max() * 1.1

    for d_source, col, title, color, (r, c) in plot_map:
        ax = axes[r, c]
        
        # Erstelle Bars
        # Wir nutzen Seed als String für die X-Achse, damit die Abstände gleichmäßig sind
        x_labels = [f"S-{int(s)}" for s in d_source['Seed']]
        bars = ax.bar(x_labels, d_source[col], color=color, edgecolor='black', alpha=0.8)
        
        # Werte über die Bars schreiben
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max_dp*0.01 if r==0 else max_ps*0.01),
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Achsen-Limits setzen für direkten Vergleich
        if r == 0: ax.set_ylim(0, max_dp)
        else:      ax.set_ylim(0, max_ps)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(col, fontweight='bold')
        ax.set_xlabel("Random Seed Number", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle("Individual Run Analysis: P0 (Baseline) vs. P1 (Optimized)\nSorted by Random Seed", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = OUT_DIR / "P0_vs_P1_2x2_Individual_Bars.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Barchart gespeichert unter: {save_path.name}")
    plt.show()

# =====================================================
# 3. START
# =====================================================
if __name__ == "__main__":
    plot_2x2_bars(df)