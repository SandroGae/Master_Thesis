import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# KONFIGURATION & PFADE
# ==========================================
DATA_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\test_4_materials")
OUTPUT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\codes")

MATERIALS = ["bamboo", "carbon_fiber", "glass_fiber", "chicken_liver"]
EXPOSURES = ["0.0001", "0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01", "0.05", "0.15"]
SLICE_IDX = 9  # Index 9 = Das 10. Bild

# ==========================================
# KONTROLLE DER HELLIGKEIT (PERCENTILES)
# ==========================================
# Format: "material_name": (unteres_perzentil, oberes_perzentil)
# Beispiel (1.0, 99.0): Ignoriert das 1% der dunkelsten und 1% der hellsten Pixel
CLIP_PERCENTILES = {
    "bamboo": (1.0, 99.0),
    "carbon_fiber": (1.0, 99.0),
    "glass_fiber": (1.0, 99.0),
    "chicken_liver": (2.0, 98.0)  # Leber braucht evtl. etwas mehr Kontrast-Schnitt
}

def main():
    print("🎨 Starte Visualisierung mit Percentile-Clipping...")

    for mat in MATERIALS:
        print(f"Verarbeite {mat}...")
        images = {}
        
        # 1. Alle Belichtungszeiten laden
        for exp in EXPOSURES:
            file_path = DATA_DIR / f"{mat}_{exp}s_test_series.npz"
            
            if file_path.exists():
                with np.load(file_path) as data:
                    vol = data['volume']
                    if vol.shape[0] > SLICE_IDX:
                        images[exp] = vol[SLICE_IDX]
                    else:
                        images[exp] = np.zeros((512, 512))
            else:
                images[exp] = np.zeros((512, 512))

        # 2. Plotting Setup
        fig = plt.figure(figsize=(14, 18))
        fig.suptitle(f"Modell-Vorhersagen: {mat.replace('_', ' ').title()} (Slice {SLICE_IDX + 1})", 
                     fontsize=18, fontweight='bold', y=0.96)

        gs = plt.GridSpec(4, 3, figure=fig, hspace=0.15, wspace=0.05)
        p_low, p_high = CLIP_PERCENTILES[mat]

        # --- Hilfsfunktion für das Zeichnen mit Perzentilen ---
        def plot_img(ax, img, title, color='black', fontsize=10):
            # Berechne die echten Werte für das Clipping basierend auf den Perzentilen
            vmin = np.percentile(img, p_low)
            vmax = np.percentile(img, p_high)
            
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=fontsize, fontweight='bold' if color != 'black' else 'normal', color=color)
            ax.axis('off')

        # --- ZEILE 1: LC (0.0001s) und GT (0.15s) ---
        ax_lc = fig.add_subplot(gs[0, 0])
        plot_img(ax_lc, images[EXPOSURES[0]], f"Low Count Prediction\n({EXPOSURES[0]}s)", 'red', 11)

        ax_explain = fig.add_subplot(gs[0, 1])
        ax_explain.text(0.5, 0.5, f"Vergleich:\nLC vs. GT\n\nClipping:\n{p_low}% - {p_high}%", 
                       fontsize=12, fontweight='bold', ha='center', va='center')
        ax_explain.axis('off')

        ax_gt = fig.add_subplot(gs[0, 2])
        plot_img(ax_gt, images[EXPOSURES[-1]], f"Ground Truth Prediction\n({EXPOSURES[-1]}s)", 'green', 11)

        # --- ZEILEN 2 bis 4: Das 3x3 Raster ---
        for i, exp in enumerate(EXPOSURES):
            row = 1 + (i // 3)
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            plot_img(ax, images[exp], f"{exp}s")

        # 3. Speichern
        out_file = OUTPUT_DIR / f"visualisierung_{mat}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✅ Gespeichert: {out_file.name}")

    print("\n🎉 Alle Visualisierungen abgeschlossen!")

if __name__ == "__main__":
    main()