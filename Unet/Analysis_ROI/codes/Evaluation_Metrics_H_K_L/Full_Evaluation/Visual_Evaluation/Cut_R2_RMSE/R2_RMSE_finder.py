import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# 1. SETUP & PFADE
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv" 
OUT_DIR = SCRIPT_DIR / "Sandra_Test_Results"
OUT_DIR.mkdir(exist_ok=True)

CUT_PERCENT = 2.0 

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {CSV_FILE}")

# =====================================================
# 2. DATEN LADEN & RATIOS BERECHNEN
# =====================================================
df = pd.read_csv(CSV_FILE)

# Ratios berechnen und Unendlichkeiten abfangen
for d in ["H", "K", "L"]:
    df[f"R2_Ratio_{d}"] = df[f"R2_pr_{d}"] / df[f"R2_gt_{d}"]
    df[f"RMSE_Ratio_{d}"] = df[f"RMSE_pr_{d}"] / df[f"RMSE_gt_{d}"]
    
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Melten und Richtungen (H, K, L) vereinheitlichen
id_vars = ['Point', 'Seed', 'SeriesID', 'Alpha', 'Beta']

def get_clean_melt(dataframe, value_cols, new_name):
    melted = dataframe.melt(id_vars=id_vars, value_vars=value_cols, var_name='Dir', value_name=new_name)
    # Extrahiere nur den letzten Buchstaben (H, K oder L)
    melted['Dir'] = melted['Dir'].str[-1]
    return melted

r2_melted = get_clean_melt(df, ['R2_Ratio_H', 'R2_Ratio_K', 'R2_Ratio_L'], 'R2_Ratio')
rmse_melted = get_clean_melt(df, ['RMSE_Ratio_H', 'RMSE_Ratio_K', 'RMSE_Ratio_L'], 'RMSE_Ratio')

# Jetzt klappt der Merge, da 'Dir' in beiden DFs nur noch 'H', 'K' oder 'L' ist
data_combined = pd.merge(r2_melted, rmse_melted, on=id_vars + ['Dir']).dropna(subset=['R2_Ratio', 'RMSE_Ratio'])

if data_combined.empty:
    print("\n[FEHLER] Keine gemeinsamen Datenpunkte nach dem Merge gefunden!")
    exit()

# =====================================================
# 3. STATISTISCHE AUSWERTUNG
# =====================================================
r2_threshold = np.percentile(data_combined['R2_Ratio'], CUT_PERCENT)
rmse_threshold = np.percentile(data_combined['RMSE_Ratio'], 100 - CUT_PERCENT)

total_potential = len(df) * 3
valid_count = len(data_combined)

print("\n" + "="*70)
print(f"{'STATISTISCHE ÜBERSICHT (Sandra-Test)':^70}")
print("="*70)
print(f"Gültige Fits (non-NaN):         {valid_count:>8} / {total_potential}")
print(f"Erfolgsquote:                   {(valid_count/total_potential)*100:>7.2f}%")
print("-" * 70)
print(f"Empfohlener R2_RATIO_MIN:       {r2_threshold:>8.4f}")
print(f"Empfohlener RMSE_RATIO_MAX:      {rmse_threshold:>8.4f}")
print("="*70)

# =====================================================
# 4. ANALYSE DER VERWÜRFE PRO SERIE
# =====================================================
data_combined['is_fail'] = (data_combined['R2_Ratio'] < r2_threshold) | \
                           (data_combined['RMSE_Ratio'] > rmse_threshold)

fail_report = data_combined.groupby('SeriesID').agg(
    Total_Fits=('is_fail', 'count'),
    Fails=('is_fail', 'sum')
).reset_index()

fail_report['Fail_Rate_%'] = (fail_report['Fails'] / fail_report['Total_Fits']) * 100

print(f"\n{'VERWÜRFE PRO SERIE (Cut: ' + str(CUT_PERCENT) + '%):':^70}")
print("-" * 70)
header = f"{'Serie':<10} | {'Fits gesamt':<15} | {'Verworfen':<12} | {'Quote (%)':<10}"
print(header)
print("-" * len(header))

for _, row in fail_report.sort_values('Fails', ascending=False).iterrows():
    print(f"S-{int(row['SeriesID']):<8} | {int(row['Total_Fits']):<15} | "
          f"{int(row['Fails']):<12} | {row['Fail_Rate_%']:>9.2f}%")
print("-" * len(header))
print(f"{'TOTAL':<10} | {int(fail_report['Total_Fits'].sum()):<15} | "
      f"{int(fail_report['Fails'].sum()):<12} | {data_combined['is_fail'].mean()*100:>9.2f}%")
print("="*70)

# =====================================================
# 5. PLOTTING
# =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

sns.histplot(data_combined['R2_Ratio'], bins=100, ax=ax1, color='skyblue', kde=True)
ax1.axvline(r2_threshold, color='red', linestyle='--', label=f'Cut-off {CUT_PERCENT}%')
ax1.set_title(f"R2-Ratio (Cut: {r2_threshold:.2f})")

sns.histplot(data_combined['RMSE_Ratio'], bins=100, ax=ax2, color='salmon', kde=True)
ax2.axvline(rmse_threshold, color='red', linestyle='--', label=f'Cut-off {CUT_PERCENT}%')
ax2.set_title(f"RMSE-Ratio (Cut: {rmse_threshold:.2f})")
# Dynamisches Limit für Sichtbarkeit (99.5% Perzentil)
ax2.set_xlim(0, np.percentile(data_combined['RMSE_Ratio'], 99.5)) 

plt.tight_layout()
plt.savefig(OUT_DIR / "Ratio_Distributions_Calibration.png")
print(f"\n>>> Ergebnisse gespeichert in: {OUT_DIR}")