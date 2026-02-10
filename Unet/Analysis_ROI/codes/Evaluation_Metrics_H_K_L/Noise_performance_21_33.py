import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# =====================================================
# 1. SETUP & PFADE (Hier lag der Fehler)
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
SAVE_DIR = ROOT_DIR / "Unet/Analysis_ROI/codes/Evaluation_Metrics_H_K_L"
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Erstellt den Ordner, falls er fehlt

# Datenpunkte
noise_scores = np.array([42.6, 43.2, 43.3, 45.3, 45.6, 47.1, 47.8, 95.8, 96.5, 100.0])
perf_21 = np.array([0.92, 0.91, 0.92, 0.90, 0.89, 0.88, 0.87, 0.45, 0.42, 0.38])
perf_33 = np.array([0.84, 0.85, 0.84, 0.86, 0.85, 0.86, 0.85, 0.83, 0.82, 0.81])

# =====================================================
# 2. MODEL-DEFINITION & FIT
# =====================================================
def sigmoid(x, L, x0, k, b):
    # Logistische Funktion für den Performance-Abfall
    return L / (1 + np.exp(-k*(x-x0))) + b

# Sigmoid-Fit für Rang 21
p0_21 = [0.5, 70, -0.2, 0.4] 
popt21, _ = curve_fit(sigmoid, noise_scores, perf_21, p0=p0_21, maxfev=10000)

# Linearer Fit für Rang 33
z_33 = np.polyfit(noise_scores, perf_33, 1)
line_33 = np.poly1d(z_33)

# =====================================================
# 3. CROSSOVER BERECHNEN
# =====================================================
x_fine = np.linspace(30, 110, 1000)
y21_fine = sigmoid(x_fine, *popt21)
y33_fine = line_33(x_fine)

idx = np.argwhere(np.diff(np.sign(y21_fine - y33_fine))).flatten()
crossover_x = x_fine[idx][0] if len(idx) > 0 else 70.0

# =====================================================
# 4. PLOTTING & SPEICHERN
# =====================================================
plt.figure(figsize=(12, 7), dpi=150)

plt.scatter(noise_scores, perf_21, color='forestgreen', s=80, edgecolors='black', label='Rang 21 Daten', zorder=5)
plt.scatter(noise_scores, perf_33, color='darkorange', s=80, edgecolors='black', label='Rang 33 Daten', zorder=5)

plt.plot(x_fine, y21_fine, color='forestgreen', lw=3, label='Rang 21 Fit (Sigmoidal)')
plt.plot(x_fine, y33_fine, color='darkorange', lw=3, label='Rang 33 Fit (Linear)')

plt.axvline(crossover_x, color='red', linestyle='--', alpha=0.6)
plt.text(crossover_x+1, 0.6, f'Crossover @ {crossover_x:.1f}', color='red', fontweight='bold')
plt.axvspan(50, 95, color='gray', alpha=0.1, label='Messlücke (Gap)')

plt.title("Optimale Trigger-Strategie: Performance vs. Noise", fontsize=14, fontweight='bold')
plt.xlabel("Noise Score (0=Rein, 100=Max Rauschen)", fontsize=12)
plt.ylabel(r"Detection Power ($SBRGain \cdot AreaRatio$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower left')

# Datei speichern
SAVE_PATH = SAVE_DIR / "Performance_Noise_Plot.png"
plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)
print(f"Erfolg: Plot wurde hier gespeichert: {SAVE_PATH}")

plt.show()
print(f"Empfohlener Threshold für Smart Sampling: {crossover_x:.2f}")