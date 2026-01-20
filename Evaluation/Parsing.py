import pandas as pd
from pathlib import Path

# Pfad-Logik
base_path = Path(__file__).parent
input_file = base_path / "evaluation_results_V2.txt"

def parse_evaluation_report(file_path):
    raw_data = []
    avg_data = []
    current_dataset = None
    current_group = None 

    if not file_path.exists():
        print(f"Fehler: {file_path} nicht gefunden!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('=') or line.startswith('REPORT') or not line.strip() or 'Model Name' in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                if parts[0] and not parts[0].startswith('-'):
                    current_dataset = parts[0]
                
                if "AVERAGE OF GROUP" not in parts[1]:
                    # Gruppe extrahieren
                    current_group = parts[1].split('_fold')[0].split('__seed')[0]
                    # Sonderfall für den einen interpolierten Fold korrigieren
                    if "fold2_only" in current_group:
                        current_group = "cross_val_unet_25d_improved_V2_interpolated"
                    
                    raw_data.append({
                        'Dataset': current_dataset, 'Group': current_group,
                        'Model_Name': parts[1], 'Loss': float(parts[2]), 
                        'MAE': float(parts[3]), 'MSE': float(parts[4]), 
                        'SSIM': float(parts[5]), 'PSNR': float(parts[6])
                    })
                elif "AVERAGE OF GROUP" in parts[1]:
                    row = {'Dataset': current_dataset, 'Group': current_group}
                    metrics = ['Loss', 'MAE', 'MSE', 'SSIM', 'PSNR']
                    for i, metric in enumerate(metrics):
                        val_str = parts[i+2]
                        if '±' in val_str:
                            mean, std = val_str.split('±')
                            row[f'{metric}_mean'] = float(mean)
                            row[f'{metric}_std'] = float(std)
                    avg_data.append(row)

    # DataFrames erstellen
    df_raw = pd.DataFrame(raw_data)
    df_avg = pd.DataFrame(avg_data)

    # --- SORTIER-LOGIK: RANDOM SEED ZUERST ---
    # Wir definieren die gewünschte Reihenfolge der Gruppen
    order = [
        'random_seed_unet_25d_improved_V2',
        'cross_val_unet_25d_improved_V2',
        'no_augmentation_unet_25d_improved_V2',
        'cross_val_unet_25d_improved_V2_interpolated',
        'no_augmentation_unet_25d_improved_V2_interpolated'
    ]
    
    # Sortierung anwenden (Datensatz zuerst, dann die definierte Gruppen-Reihenfolge)
    df_raw['Group'] = pd.Categorical(df_raw['Group'], categories=order, ordered=True)
    df_avg['Group'] = pd.Categorical(df_avg['Group'], categories=order, ordered=True)
    
    df_raw = df_raw.sort_values(['Dataset', 'Group'])
    df_avg = df_avg.sort_values(['Dataset', 'Group'])

    # Speichern
    df_raw.to_csv(base_path / 'evaluation_raw.csv', index=False)
    df_avg.to_csv(base_path / 'evaluation_averages.csv', index=False)
    print(f"CSVs erfolgreich sortiert erstellt. 'Random Seed' ist jetzt an erster Stelle.")

parse_evaluation_report(input_file)