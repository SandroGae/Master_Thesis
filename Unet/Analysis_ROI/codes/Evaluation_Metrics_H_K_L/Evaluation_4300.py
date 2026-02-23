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
LOG_ROOT = Path.home() / "scratch/43_Models_10_Seeds" 
OUT_DIR  = BASE_DIR / "Final_Heatmaps"
CSV_PATH = OUT_DIR / "Full_Evaluation_Results.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [0.0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333, 1.0]
FIT_WINDOW = (140, 240) 

# =====================================================
# 2. HILFSFUNKTIONEN & LOG-CACHING
# =====================================================
print(">>> Scanne Log-Verzeichnis und indiziere Trainings-Status...")
LOG_CACHE = {}
for json_file in LOG_ROOT.rglob("*_02_meta.json"):
    match = re.search(r'(P\d+_a\d+\.\d+_b\d+\.\d+_seed\d+)', json_file.name)
    if match:
        try:
            with open(json_file, 'r') as f:
                log_data = json.load(f)
                target_epochs = log_data.get('epochs', 200)
                actual_epochs = len(log_data.get('history', {}).get('loss', []))
                is_aborted = log_data.get('aborted', True)
                
                # Definition: Sauber = Nicht abgestürzt UND vorzeitig beendet (Early Stopping)
                is_clean = (not is_aborted) and (actual_epochs < target_epochs)
                LOG_CACHE[match.group(1)] = is_clean
        except:
            continue

print(f">>> {len(LOG_CACHE)} Logs erfolgreich indiziert.")

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def perform_area_fit(x, y):
    if np.max(y) <= 0: return None
    p0 = [np.max(y), x[np.argmax(y)], 5.0]
    bounds = ([0, x[0], 0.5], [np.inf, x[-1], 20.0])
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=2000)
        return {"mu": popt[1], "area": popt[0] * popt[2]}
    except: return None

# =====================================================
# 3. DATEN-PROZESSING
# =====================================================
def run_processing():
    if CSV_PATH.exists():
        print(f">>> CSV existiert bereits: {CSV_PATH}. Lade vorhandene Daten.")
        return pd.read_csv(CSV_PATH)

    all_files = sorted(list(NPZ_DIR.glob("*.npz")))
    print(f">>> Starte Analyse von {len(all_files)} Dateien...")
    
    results = []
    for i, path in enumerate(all_files):
        match = re.search(r'Eval_(P\d+)_a(\d+\.\d+)_b(\d+\.\d+)_seed(\d+)_S(\d+)', path.stem)
        if not match: continue
        
        p_str, alpha, beta, seed, s_id = match.groups()
        model_uid = f"{p_str}_a{alpha}_b{beta}_seed{seed}"

        try:
            data = np.load(path)
            # L-Richtung (X-Achse), Slice 15
            img_pr, img_gt = np.clip(data['pred'][15], 0, None), np.clip(data['gt'][15], 0, None)
            prof_pr = np.sum(img_pr[102:117, :], axis=0)
            prof_gt = np.sum(img_gt[102:117, :], axis=0)
            x_ax = np.arange(240)

            fit_pr = perform_area_fit(x_ax[FIT_WINDOW[0]:FIT_WINDOW[1]], prof_pr[FIT_WINDOW[0]:FIT_WINDOW[1]])
            fit_gt = perform_area_fit(x_ax[FIT_WINDOW[0]:FIT_WINDOW[1]], prof_gt[FIT_WINDOW[0]:FIT_WINDOW[1]])

            if fit_pr and fit_gt:
                is_clean = LOG_CACHE.get(model_uid, False)
                results.append({
                    "Alpha": float(alpha), "Beta": float(beta), "Seed": int(seed), 
                    "AreaRatio": fit_pr['area'] / fit_gt['area'],
                    "PosShift": abs(fit_pr['mu'] - fit_gt['mu']),
                    "IsClean": is_clean
                })
        except: continue
        
        if i % 200 == 0: print(f"Fortschritt: {i}/{len(all_files)}")

    df = pd.DataFrame(results)
    df["Alpha"] = df["Alpha"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    df["Beta"] = df["Beta"].apply(lambda x: min(STEPS, key=lambda s: abs(s-x)))
    df.to_csv(CSV_PATH, index=False)
    return df

if __name__ == "__main__":
    data = run_processing()