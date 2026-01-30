import pandas as pd
from pathlib import Path

# Pfad-Logik für deinen Evaluation-Ordner
script_dir = Path(__file__).parent
csv_path = script_dir / "evaluation_averages.csv"

# Daten laden
try:
    df = pd.read_csv(csv_path)
    print(f"Daten geladen aus: {csv_path}")
except FileNotFoundError:
    print(f"Fehler: {csv_path} nicht gefunden.")
    exit()

# Metriken definieren
metrics_to_rank = {'Loss_mean': False, 'SSIM_mean': True, 'PSNR_mean': True}

rankings = []
for ds in df['Dataset'].unique():
    ds_data = df[df['Dataset'] == ds].copy()
    for metric, higher_is_better in metrics_to_rank.items():
        # Rang berechnen: 1 ist das beste Modell für dieses spezifische Datenset
        ds_data[f'Rank_{metric}'] = ds_data[metric].rank(ascending=not higher_is_better, method='min')
    rankings.append(ds_data)

df_ranked = pd.concat(rankings)

# Finale Auswertung
# Wir aggregieren die Ränge und die Standardabweichungen (für die Robustheit)
final_ranking = df_ranked.groupby('Group').agg({
    'Rank_Loss_mean': 'mean',
    'Rank_SSIM_mean': 'mean',
    'Rank_PSNR_mean': 'mean',
    'SSIM_std': 'mean',
    'PSNR_std': 'mean'
})

# Den Overall Rank berechnen (Durchschnitt aus den drei Metrik-Rängen)
final_ranking['Overall_Rank'] = final_ranking[['Rank_Loss_mean', 'Rank_SSIM_mean', 'Rank_PSNR_mean']].mean(axis=1)

# Spalten umbenennen für bessere Lesbarkeit
final_ranking.columns = ['Avg_Rank_Loss', 'Avg_Rank_SSIM', 'Avg_Rank_PSNR', 'Avg_SSIM_Std', 'Avg_PSNR_Std', 'Overall_Rank']

print("\n--- Modell-Ranking (Niedrigerer Rank ist besser) ---")
# Sortieren nach Overall_Rank
result = final_ranking.sort_values('Overall_Rank')
print(result[['Overall_Rank', 'Avg_SSIM_Std', 'Avg_PSNR_Std']])