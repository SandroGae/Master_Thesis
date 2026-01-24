import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURATION ---
# Pfad zu deinem Ordner mit den CSVs
CSV_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\CSV\csv_triple_loss")
# Speicherort für die Bilder (wird automatisch erstellt)
OUTPUT_DIR = CSV_DIR / "loss_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_losses():
    # Suche alle log-Dateien (rekursiv, falls Unterordner existieren)
    csv_files = list(CSV_DIR.glob("**/log_*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {CSV_DIR} gefunden!")
        return

    print(f"Starte Plotting für {len(csv_files)} Dateien...")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            run_name = csv_path.stem.replace("log_", "")

            # Prüfen ob die nötigen Spalten da sind
            if 'loss' not in df.columns or 'val_loss' not in df.columns:
                print(f"Überspringe {csv_path.name}: Spalten fehlen.")
                continue

            # Plot erstellen
            # Wir nutzen kein .figure(), um Speicher zu sparen bei vielen Plots
            plt.plot(df['epoch'], df['loss'], label='Training Loss', color='blue', linewidth=1.5)
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linewidth=1.5)

            plt.title(f"Convergence: {run_name}", fontsize=10)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            # Speichern
            save_path = OUTPUT_DIR / f"plot_{run_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            # WICHTIG: Den Plot leeren, damit die Kurven nicht im nächsten Bild erscheinen
            plt.clf()
            print(f"Erstellt: {save_path.name}")

        except Exception as e:
            print(f"Fehler bei {csv_path.name}: {e}")

    plt.close()
    print(f"\nFertig! Alle Plots liegen in: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_losses()