import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. SETUP & PFADE
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Raw.csv"
TXT_BEST = SCRIPT_DIR / "Full_Evaluation_BEST_Raw.txt"
TXT_AVG  = SCRIPT_DIR / "Full_Evaluation_AVERAGE_Raw.txt"

SERIES_MASTER = {
    5:  {"win_h": (2,38), "win_k": (90,130), "win_l": (140,240)},
    11: {"win_h": (2,38), "win_k": (90,130), "win_l": (43,143)},
    12: {"win_h": (2,38), "win_k": (90,130), "win_l": (24,124)},
    15: {"win_h": (2,38), "win_k": (90,130), "win_l": (98,198)},
    16: {"win_h": (2,38), "win_k": (90,130), "win_l": (76,176)},
    21: {"win_h": (2,38), "win_k": (90,130), "win_l": (140,240)},
    22: {"win_h": (2,38), "win_k": (90,130), "win_l": (134,234)},
    29: {"win_h": (2,38), "win_k": (90,130), "win_l": (20,120)},
    35: {"win_h": (2,38), "win_k": (90,130), "win_l": (64,164)},
    50: {"win_h": (2,38), "win_k": (90,130), "win_l": (52,152)},
}

if not CSV_FILE.exists():
    print(f"Fehler: {CSV_FILE} fehlt!")
    exit()

df = pd.read_csv(CSV_FILE)

# =====================================================
# 2. VALIDIERUNG (DER "SIEB"-PROZESS)
# =====================================================
def validate_and_calculate(row):
    s_id = int(row['SeriesID'])
    cfg = SERIES_MASTER.get(s_id)
    res = {}
    row_rejections = 0
    
    for d in ["H", "K", "L"]:
        a_pr, s_pr, m_pr = row[f"Amp_pr_{d}"], row[f"Sig_pr_{d}"], row[f"Mu_pr_{d}"]
        a_gt, s_gt, m_gt = row[f"Amp_gt_{d}"], row[f"Sig_gt_{d}"], row[f"Mu_gt_{d}"]
        
        win = cfg[f"win_{d.lower()}"]
        win_w = win[1] - win[0]
        
        is_invalid = False
        # 1. SNR Check: Zu schwach für CDW-Erkennung?
        if np.isnan(a_pr) or a_pr < 0.05: is_invalid = True
        # 2. Window Check
        elif (m_pr < win[0]) or (m_pr > win[1]): is_invalid = True
        # 3. Position Check (2-Sigma)
        elif np.abs(m_pr - m_gt) > (2.0 * s_gt): is_invalid = True
        # 4. Width Check (4-Sigma)
        elif (s_pr > (4.0 * s_gt)) or (s_pr > (win_w / 2.0)): is_invalid = True
        # 5. Artefakt Check
        elif (s_pr < 0.5): is_invalid = True

        if is_invalid:
            v_amp, v_sig = 0.0, 0.0
            row_rejections += 1
        else:
            v_amp, v_sig = a_pr, s_pr
            
        area_pr, area_gt = v_amp * v_sig, a_gt * s_gt
        res[f"AreaRatio_{d}"] = area_pr / (area_gt + 1e-9)
        res[f"SBRRatio_{d}"]  = v_amp / (a_gt + 1e-9)
        res[f"FinalShift_{d}"] = np.abs(m_pr - m_gt) if not np.isnan(m_pr) else 100.0

    res['AreaRatio_3D'] = (res[f"AreaRatio_H"] + res[f"AreaRatio_K"] + res[f"AreaRatio_L"]) / 3.0
    res['SBRRatio_3D']  = (res[f"SBRRatio_H"] + res[f"SBRRatio_K"] + res[f"SBRRatio_L"]) / 3.0
    res['Shift_3D']      = (res[f"FinalShift_H"] + res[f"FinalShift_K"] + res[f"FinalShift_L"]) / 3.0
    res['RejectedCount'] = row_rejections
    return pd.Series(res)

print(">>> Verarbeite Daten...")
df_metrics = df.apply(validate_and_calculate, axis=1)
df = pd.concat([df, df_metrics], axis=1)

# ======================================================================
# TEIL 1: RANKING NACH SBRRATIO (KONTRAST)
# ======================================================================
run_ranking = df.groupby(['Point', 'Seed', 'Alpha', 'Beta']).agg(
    MeanAreaRatio=('AreaRatio_3D', 'mean'),
    MeanSBRRatio=('SBRRatio_3D', 'mean'),
    MeanShift=('Shift_3D', 'mean'),
    TotalRejected=('RejectedCount', 'sum'),
    IsClean=('IsClean', 'all')
).reset_index()

run_ranking['RejRate'] = (run_ranking['TotalRejected'] / 30.0) * 100
# Wir sortieren nach SBRRatio, da dies für CDW-Erkennung (Kontrast) wichtiger ist
all_runs_sorted = run_ranking.sort_values('MeanSBRRatio', ascending=False).reset_index(drop=True)

with open(TXT_BEST, "w", encoding="utf-8") as f:
    f.write("="*115 + "\n")
    f.write("BEST RUN RANKING (Sorted by Mean SBRRatio / Contrast Recovery)\n")
    f.write("-" * 115 + "\n")
    header = f"{'Rank':<6} {'Point':<7} {'Seed':<6} {'Alpha':<8} {'Beta':<8} {'AreaRatio':<12} {'SBRRatio':<10} {'AvgShift':<10} {'RejRate':<10} {'Clean'}\n"
    f.write(header)
    f.write("-" * 115 + "\n")
    for i, row in all_runs_sorted.iterrows():
        clean_str = "YES" if row.IsClean else "NO"
        f.write(f"{i+1:<6} P{int(row.Point):02d} {int(row.Seed):<6} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                f"{row.MeanAreaRatio:<12.4f} {row.MeanSBRRatio:<10.4f} {row.MeanShift:<10.2f} "
                f"{row.RejRate:>8.1f}% {clean_str:>7}\n")

# ======================================================================
# TEIL 2: DURCHSCHNITT DER PUNKTE
# ======================================================================
df_avg = df.copy()
mask_a1 = (df_avg["Alpha"] == 1.0); df_avg.loc[mask_a1, "Beta"] = 0.0

point_stats = df_avg.groupby(['Point', 'Alpha', 'Beta']).agg(
    Mean_AreaRatio_3D=('AreaRatio_3D', 'mean'),
    Std_AreaRatio_3D=('AreaRatio_3D', 'std'),
    Mean_SBRRatio_3D=('SBRRatio_3D', 'mean'),
    Mean_Shift_3D=('Shift_3D', 'mean'),
    Avg_Rejected=('RejectedCount', 'mean'),
    Total_Entries=('IsClean', 'count'),
    Success_Entries=('IsClean', lambda x: x.sum())
).reset_index()

point_stats['RejRate'] = (point_stats['Avg_Rejected'] / 3.0) * 100
point_stats['SuccessRate'] = (point_stats['Success_Entries'] / point_stats['Total_Entries']) * 100
# Auch hier Sortierung nach Kontrast-Wiederherstellung (SBR)
point_ranking = point_stats.sort_values('Mean_SBRRatio_3D', ascending=False).reset_index(drop=True)

with open(TXT_AVG, "w", encoding="utf-8") as f:
    f.write("="*130 + "\n")
    f.write("HYPERPARAMETER POINT AVERAGES (Ranked by Mean SBRRatio)\n")
    f.write("-" * 130 + "\n")
    header = f"{'Rank':<6} {'Point':<7} {'Alpha':<8} {'Beta':<8} {'MeanAreaRatio':<15} {'SBRRatio':<10} {'StdDev':<10} {'AvgShift':<10} {'RejRate':<12} {'Success'}\n"
    f.write(header)
    f.write("-" * 130 + "\n")
    for i, row in point_ranking.iterrows():
        f.write(f"{i+1:<6} P{int(row.Point):02d} {row.Alpha:<8.4f} {row.Beta:<8.4f} "
                f"{row.Mean_AreaRatio_3D:<15.4f} {row.Mean_SBRRatio_3D:<10.4f} {row.Std_AreaRatio_3D:<10.4f} "
                f"{row.Mean_Shift_3D:<10.2f} {row.RejRate:>10.1f}% {row.SuccessRate:>9.1f}%\n")

print(f"Ranking nach SBRRatio abgeschlossen.")