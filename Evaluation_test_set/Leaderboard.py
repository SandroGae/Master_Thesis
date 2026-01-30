import pandas as pd
from pathlib import Path

# --- PFAD-FIX ---
# Script liegt in: Evaluation_test_set
# CSVs liegen in: Evaluation_test_set/CSV_Evaluation_test_set
base_path = Path(__file__).parent
csv_path = base_path / "CSV_Evaluation_test_set" / "evaluation_averages.csv"

# Daten laden
try:
    df = pd.read_csv(csv_path)
    print(f"Daten geladen aus: {csv_path}")
except FileNotFoundError:
    print(f"Fehler: {csv_path} nicht gefunden. Hast du das kombinierte Skript schon laufen lassen?")
    exit()

# Metriken definieren
metrics_to_rank = {'Loss_mean': False, 'SSIM_mean': True, 'PSNR_mean': True}

rankings = []
for ds in df['Dataset'].unique():
    ds_data = df[df['Dataset'] == ds].copy()
    for metric, higher_is_better in metrics_to_rank.items():
        # Rang berechnen
        ds_data[f'Rank_{metric}'] = ds_data[metric].rank(ascending=not higher_is_better, method='min')
    rankings.append(ds_data)

df_ranked = pd.concat(rankings)

# Finale Auswertung
final_ranking = df_ranked.groupby('Group').agg({
    'Rank_Loss_mean': 'mean',
    'Rank_SSIM_mean': 'mean',
    'Rank_PSNR_mean': 'mean',
    'SSIM_std': 'mean',
    'PSNR_std': 'mean'
})

# Overall Rank berechnen
final_ranking['Overall_Rank'] = final_ranking[['Rank_Loss_mean', 'Rank_SSIM_mean', 'Rank_PSNR_mean']].mean(axis=1)

# Spalten umbenennen
final_ranking.columns = ['Avg_Rank_Loss', 'Avg_Rank_SSIM', 'Avg_Rank_PSNR', 'Avg_SSIM_Std', 'Avg_PSNR_Std', 'Overall_Rank']

print("\n--- MODELL-RANKING (Niedrigerer Overall_Rank ist besser) ---")
result = final_ranking.sort_values('Overall_Rank')
print(result[['Overall_Rank', 'Avg_SSIM_Std', 'Avg_PSNR_Std']])