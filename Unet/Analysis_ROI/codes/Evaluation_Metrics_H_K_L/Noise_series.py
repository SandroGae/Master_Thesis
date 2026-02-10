import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================
# 1. SETUP & CONFIG
# =====================================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS")
IN_DIR = ROOT_DIR / "Unet/Analysis_ROI/Prediction.npz/Predictions_Raw_new_RERUN"

# Wir nehmen ein Referenz-Modell, um an die LC-Daten zu kommen (LC ist bei allen gleich)
REF_ID = "a0.0000_b0.0000_seed43" 

SERIES_CONFIG = {
    5:  {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    11: {"roi_y": (100, 119), "bg_gap": 5, "bg_h": 10},
    12: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    15: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    16: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    21: {"roi_y": (101, 118), "bg_gap": 5, "bg_h": 10},
    22: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    29: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    35: {"roi_y": (102, 117), "bg_gap": 5, "bg_h": 10},
    50: {"roi_y": (102, 117), "bg_gap": 10, "bg_h": 10},
}

# =====================================================
# 2. NOISE BERECHNUNG
# =====================================================
def calculate_series_noise(s_id, cfg):
    file_path = IN_DIR / f"Pred_DeepScan_{REF_ID}_D5_S{s_id}_FullSeries.npz"
    if not file_path.exists():
        return None
    
    # lc laden (Low Count) [cite: 2026-02-05]
    data = np.load(file_path)
    lc = data['lc'] # Shape (41, 192, 240) oder (41, Z, 192, 240)
    
    # Hintergrund-Bereiche definieren (wie in deinem Analyse-Code)
    ry, gap, h = cfg["roi_y"], cfg["bg_gap"], cfg["bg_h"]
    r1_t, r1_b = max(0, ry[0] - gap - h), max(0, ry[0] - gap)
    r2_t, r2_b = min(192, ry[1] + gap), min(192, ry[1] + gap + h)
    
    # Wir mitteln das Rauschen über alle 41 Frames
    noise_values = []
    for f in range(41):
        img = lc[f, 0] if lc.ndim == 4 else lc[f]
        
        # Pixel-Werte aus den zwei Hintergrund-Streifen extrahieren
        bg_pixels = np.concatenate([img[r1_t:r1_b, :].flatten(), 
                                    img[r2_t:r2_b, :].flatten()])
        
        # Metrik: Standardabweichung normiert auf die mittlere Intensität (Poisson-Noise-Schätzung)
        if np.mean(bg_pixels) > 0:
            # CV = Standardabweichung / Mittelwert
            noise_metric = np.std(bg_pixels) / np.sqrt(np.mean(bg_pixels))
            noise_values.append(noise_metric)
            
    return np.mean(noise_values)

# =====================================================
# 3. AUSWERTUNG & SCALING (0-100)
# =====================================================
results = []
for s_id, cfg in SERIES_CONFIG.items():
    n_val = calculate_series_noise(s_id, cfg)
    if n_val:
        results.append({"Serie": s_id, "RawNoise": n_val})

df_noise = pd.DataFrame(results)

# Skalierung: 0 = kein Rauschen, 100 = Maximum der gefundenen Werte
# Wir nutzen ein festes Scaling (z.B. Faktor 1.5), damit 100 wirklich "extrem" ist
max_observed = df_noise['RawNoise'].max()
df_noise['Noise_Score'] = (df_noise['RawNoise'] / max_observed) * 100
df_noise['Noise_Score'] = df_noise['Noise_Score'].round(1)

print("\n--- NOISE LEVEL EVALUATION (0-100) ---")
print(df_noise[['Serie', 'Noise_Score']].sort_values('Noise_Score', ascending=False).to_string(index=False))

# Optional: Als CSV speichern für das Gating-Modell
# df_noise.to_csv(ROOT_DIR / "Noise_Levels_S_Series.csv", index=False)