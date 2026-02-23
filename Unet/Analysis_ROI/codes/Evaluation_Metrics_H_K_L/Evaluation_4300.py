import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path
import re
import json
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. KONFIGURATION & PFADE
# =====================================================
BASE_DIR = Path.home() / "scratch/Evaluation_Pipeline"
NPZ_DIR  = BASE_DIR / "Evaluation_results"
# Pfad zu deinen Trainings-Logs (JSON), um Early-Stopping zu prüfen
LOG_DIR  = Path.home() / "scratch/Confidence/logs" 
OUT_DIR  = BASE_DIR / "Final_Heatmaps"
CSV_PATH = OUT_DIR / "Full_Evaluation_Results.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 7x7 Raster Definition
STEPS = [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 1.0]

# Wir nehmen die L-Richtung Parameter als Referenz für den Fit
FIT_WINDOW = (140, 240) 

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_area_fit(x, y):
    """Extrahiert Amplitude und Sigma via Gauss-Fit"""
    p0 = [np.max(y), x[np.argmax(y)], 5.0]
    bounds = ([0, x[0], 0.5], [np.inf, x[-1], 20.0])
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=2000)
        # Area = Amplitude * Sigma * sqrt(2*pi) -> sqrt(2pi) ist konstant, weglassen für Ratio
        return {"amp": popt[0], "mu": popt[1], "sigma": popt[2], "area": popt[0] * popt[2]}
    except:
        return None

def check_training_status(model_id):
    """Prüft ob Modell via Early Stopping (True) oder Max Epochs/Crash (False) endete"""
    # Suche das passende JSON log
    log_files = list(LOG_DIR.glob(f"*{model_id}*.json"))
    if not log_files: return False # Im Zweifel als 'nicht sauber' markieren
    
    try:
        with open(log_files[0], 'r') as f:
            log_data = json.load(f)
            # Logik: Wenn gestoppte Epoche < Max Epochen -> Early Stopping Erfolg
            return log_data.get('stopped_epoch', 999) < log_data.get('epochs', 1000)
    except:
        return False

# =====================================================
# 3. DATEN-PROZESSING (CSV GENERIERUNG)
# =====================================================
def run_processing():
    if CSV_PATH.exists():
        print(f">>> CSV existiert bereits: {CSV_PATH}. Überspringe Berechnung.")
        return pd.read_csv(CSV_PATH)

    all_files = list(NPZ_DIR.glob("*.npz"))
    print(f">>> Starte Analyse von {len(all_files)} Dateien...")
    
    results = []
    
    for i, path in enumerate(all_files):
        # Extrahiere Metadaten aus Dateiname
        match = re.search(r'P(\d+)_a(\d+\.\d+)_b(\d+\.\d+)_seed(\d+)_S(\d+)', path.stem)
        if not match: continue
        
        p_idx, alpha, beta, seed, s_id = match.groups()
        model_uid = f"a{alpha}_b{beta}_seed{seed}" # ID um das Training-Log zu finden

        data = np.load(path)
        # Wir nutzen das Profil in L-Richtung (X-Achse) für die Fläche
        # Slice Index 15 als Standard aus deinem L-Skript
        img_pr = data['pred'][15]
        img_gt = data['gt'][15]
        
        # Profile extrahieren (ROI y: 102-117)
        prof_pr = np.sum(img_pr[102:117, :], axis=0)
        prof_gt = np.sum(img_gt[102:117, :], axis=0)
        x_ax = np.arange(240)

        fit_pr = perform_area_fit(x_ax[FIT_WINDOW[0]:FIT_WINDOW[1]], prof_pr[FIT_WINDOW[0]:FIT_WINDOW[1]])
        fit_gt = perform_area_fit(x_ax[FIT_WINDOW[0]:FIT_WINDOW[1]], prof_gt[FIT_WINDOW[0]:FIT_WINDOW[1]])

        if fit_pr and fit_gt:
            is_clean = check_training_status(model_uid)
            results.append({
                "Alpha": float(alpha), "Beta": float(beta), "Seed": int(seed), "Series": int(s_id),
                "AreaRatio": fit_pr['area'] / fit_gt['area'], # GT ist hier implizit normiert
                "PosShift": abs(fit_pr['mu'] - fit_gt['mu']),
                "IsClean": is_clean
            })
        
        if i % 200 == 0: print(f"Fortschritt: {i}/{len(all_files)}")

    df = pd.DataFrame(results)
    # Runden auf offizielle Gitter-Werte
    df["Alpha"] = df["Alpha"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    df["Beta"] = df["Beta"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    
    df.to_csv(CSV_PATH, index=False)
    return df

# =====================================================
# 4. PLOTTING (HEATMAPS)
# =====================================================
def plot_heatmaps(df):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # --- PLOT 1: Area All (Alle 100 NPZs) ---
    pivot_all = df.groupby(['Alpha', 'Beta'])['AreaRatio'].mean().unstack()
    sns.heatmap(pivot_all, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[0])
    axes[0].set_title("1. Avg Area Ratio (All Runs)\nGT Normalized to 1.0")

    # --- PLOT 2: Area Clean (Only Early Stopping) + Errorbars via STD ---
    clean_df = df[df['IsClean'] == True]
    pivot_clean = clean_df.groupby(['Alpha', 'Beta'])['AreaRatio'].mean().unstack()
    
    # Berechnung der "Unzuverlässigkeit": 1 - (Anzahl Clean / Soll-Anzahl)
    # Je weniger sauber, desto dunkler die Markierung/größer der theoretische Fehler
    count_ratio = df.groupby(['Alpha', 'Beta'])['IsClean'].mean().unstack()
    
    sns.heatmap(pivot_clean, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1])
    # Visueller Hinweis auf unsaubere Runs (Dots werden kleiner bei Ausfall)
    axes[1].set_title("2. Avg Area Ratio (Early Stopping Only)\nValues: Clean Runs Mean")

    # --- PLOT 3: Positional Shift (Clean Only) ---
    pivot_shift = clean_df.groupby(['Alpha', 'Beta'])['PosShift'].mean().unstack()
    sns.heatmap(pivot_shift, annot=True, fmt=".2f", cmap="Reds_r", ax=axes[2])
    axes[2].set_title("3. Avg Positional Shift (Clean Only)\nLower is Better (Pixels)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Final_Evaluation_Heatmaps.png", dpi=150)
    print(f">>> Plots gespeichert in {OUT_DIR}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    data = run_processing()
    plot_heatmaps(data)