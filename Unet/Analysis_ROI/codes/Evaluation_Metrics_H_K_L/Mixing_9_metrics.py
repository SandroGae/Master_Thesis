import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
CSV_FILE = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/CDW_Hyperparameter_Results.csv"
OUT_DIR  = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L/Hyperparameter_Topologies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# =====================================================
# 2. BERECHNUNG & NORMIERUNG
# =====================================================
# A. Neue Metrik: Detection Power
df['DetectionPower'] = df['SBRGain'] * df['AreaRatio']

# B. Mittelung über H, K, L
model_avg = df.groupby(['Alpha', 'Beta']).agg({
    'PosShift': 'mean',
    'DetectionPower': 'mean'
}).reset_index()

# C. Normierung für fairen Vergleich (0 = Schlecht, 1 = Bestmöglich)
# Für Power: Höher ist besser
p_min, p_max = model_avg['DetectionPower'].min(), model_avg['DetectionPower'].max()
model_avg['Power_Norm'] = (model_avg['DetectionPower'] - p_min) / (p_max - p_min)

# Für Shift: Niedriger ist besser -> wir invertieren, damit 1 = "kein Shift"
s_min, s_max = model_avg['PosShift'].min(), model_avg['PosShift'].max()
model_avg['Shift_Norm'] = 1.0 - ((model_avg['PosShift'] - s_min) / (s_max - s_min))

# D. Kombinierter Score (50/50 gewichtet)
model_avg['Trigger_Reliability'] = (model_avg['Power_Norm'] + model_avg['Shift_Norm']) / 2

# =====================================================
# 3. PLOTTING (1x2 Heatmap Layout)
# =====================================================
def plot_trigger_analysis(data):
    # Wir plotten die Originalwerte, nutzen aber RdYlGn für die Bewertung
    metrics = [
        ('PosShift', 'RdYlGn_r', 'Precision: Avg Pos Shift (Lower = Green)'),
        ('DetectionPower', 'RdYlGn', 'Visibility: Detection Power (Higher = Green)')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    plt.subplots_adjust(wspace=0.25)

    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xi, yi)

    for i, (m_col, cmap, m_title) in enumerate(metrics):
        ax = axes[i]
        
        # Interpolation der Topologie
        Z = griddata((data['Alpha'], data['Beta']), data[m_col], (X, Y), method='cubic')
        
        # Plot
        cp = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        cb = fig.colorbar(cp, ax=ax)
        cb.set_label(m_col, fontweight='bold')
        
        ax.contour(X, Y, Z, levels=12, colors='black', alpha=0.1)
        ax.scatter(data['Alpha'], data['Beta'], c='white', edgecolors='black', s=30, alpha=0.7)
        
        # Achsenbeschriftung
        ax.set_title(m_title, fontsize=13, fontweight='bold')
        ax.set_xlabel(r"$\alpha$ (alpha * SSIM)", fontsize=11)
        if i == 0:
            ax.set_ylabel(r"$\beta$ (beta * MSE + (1 - beta) * MAE)", fontsize=11)

    plt.suptitle("Trigger Optimization Strategy: Precision vs. Detection Strength", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    save_path = OUT_DIR / "Heatmap_Dual_Trigger_Analysis.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# =====================================================
# 4. START & RANKING AUSGABE
# =====================================================
plot_trigger_analysis(model_avg)

print("\nTOP 5 MODELLE FÜR SMART SAMPLING (Balanced Score):")
top_models = model_avg.sort_values('Trigger_Reliability', ascending=False).head(5)
print(top_models[['Alpha', 'Beta', 'PosShift', 'DetectionPower', 'Trigger_Reliability']].to_string(index=False))