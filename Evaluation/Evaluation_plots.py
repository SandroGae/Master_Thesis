import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Pfad-Logik
script_dir = Path(__file__).parent
csv_path = script_dir / "evaluation_averages.csv"
output_dir = script_dir / "Evaluation_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# Daten laden
df_avg = pd.read_csv(csv_path)

# --- MAPPING & REIHENFOLGE ---
group_mapping = {
    'random_seed_unet_25d_improved_V2': 'Seed Test',
    'cross_val_unet_25d_improved_V2_interpolated': 'Aug/Int',
    'no_augmentation_unet_25d_improved_V2_interpolated': 'No Aug/Int',
    'cross_val_unet_25d_improved_V2': 'Aug/No Int',
    'no_augmentation_unet_25d_improved_V2': 'No Aug/No Int'
}

plot_order = ['Seed Test', 'Aug/Int', 'No Aug/Int', 'Aug/No Int', 'No Aug/No Int']

# Labels anwenden und Sortierung erzwingen
df_avg['Display_Name'] = df_avg['Group'].map(group_mapping)
df_avg['Display_Name'] = pd.Categorical(df_avg['Display_Name'], categories=plot_order, ordered=True)
df_avg = df_avg.sort_values('Display_Name')

# Metriken und Farben
metrics = ['Loss', 'MAE', 'MSE', 'SSIM', 'PSNR']
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plot_order)))
color_map = dict(zip(plot_order, colors))

# --- GLOBALE SKALIERUNG BERECHNEN ---
# Wir berechnen die Limits einmal über das gesamte DataFrame, damit alle Plots gleich sind
global_limits = {}
for metric in metrics:
    m_mean = df_avg[f'{metric}_mean']
    m_std = df_avg[f'{metric}_std']
    
    # Untere Grenze (Min Mean - Max Std) und obere Grenze (Max Mean + Max Std)
    # Ein kleiner Puffer von 5% wird hinzugefügt
    lower = (m_mean - m_std).min()
    upper = (m_mean + m_std).max()
    
    if metric == 'SSIM':
        global_limits[metric] = (lower * 0.99, min(1.0, upper * 1.01))
    elif metric == 'PSNR':
        global_limits[metric] = (lower - 1.0, upper + 1.0)
    else: # Loss, MAE, MSE starten bei 0 für bessere Vergleichbarkeit
        global_limits[metric] = (0, upper * 1.2)

# --- PLOTTEN PRO DATENSET ---
for ds in df_avg['Dataset'].unique():
    ds_data = df_avg[df_avg['Dataset'] == ds]
    
    fig, axes = plt.subplots(1, 5, figsize=(26, 8), constrained_layout=True)
    fig.suptitle(f'Modell-Performance (Synchronisierte Skala) - Datenset: {ds}', 
                 fontsize=22, fontweight='bold', y=1.05)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        labels = ds_data['Display_Name']
        means = ds_data[f'{metric}_mean']
        stds = ds_data[f'{metric}_std']
        
        # Balken zeichnen
        ax.bar(labels, means, yerr=stds, color=[color_map[l] for l in labels], 
               capsize=8, alpha=0.9, edgecolor='black', linewidth=1.5)
        
        ax.set_title(metric, fontsize=18, pad=15, fontweight='semibold')
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # --- FIXIERTE Y-ACHSE ANWENDEN ---
        ax.set_ylim(global_limits[metric])

    # Speichern
    file_name = output_dir / f"plot_{ds.replace('.', '_')}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Synchronisierter Plot erstellt: {file_name}")

print("\n--- Alle Plots wurden mit identischen Achsen-Skalen erstellt! ---")