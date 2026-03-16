import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# SETUP
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\Test_replication_V2_triple_loss_test")
OUT_ROOT = BASE_DIR / "Rekonstruktionen_Test"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Wir wählen 5 repräsentative Slices aus dem Bereich 2-37
SELECTED_SLICES = [5, 12, 19, 26, 33]

def plot_selected_slices():
    all_files = sorted(list(BASE_DIR.rglob("*.npz")))
    
    if not all_files:
        print(f"❌ Keine .npz Dateien gefunden in {BASE_DIR}")
        return

    print(f"🔍 Starte Visualisierung von {len(all_files)} Modellen mit je 5 Slices...")

    for f_path in all_files:
        point_folder = f_path.parent.name 
        model_name = f_path.stem
        
        # Unterordner-Struktur: Rekonstruktionen_Test/P02/Modellname/
        series_out = OUT_ROOT / point_folder / model_name
        series_out.mkdir(parents=True, exist_ok=True)
        
        print(f"⌛ Plotting: {point_folder} -> {model_name}")
        
        data = np.load(f_path)
        lc, pred, gt = data['lc'], data['pred'], data['gt']

        for idx in SELECTED_SLICES:
            p_slice, gt_slice, lc_slice = pred[idx], gt[idx], lc[idx]

            # --- FIX: Gemeinsame Limits für den visuellen Vergleich ---
            # Wir nehmen die Grenzwerte vom Ground Truth als Master-Maßstab
            vmin_shared = np.percentile(gt_slice, 1)
            vmax_shared = np.percentile(gt_slice, 99)
            
            # Für den verrauschten Input nehmen wir oft einen separaten Stretch, 
            # aber Pred und GT MÜSSEN identisch sein.
            vmin_lc, vmax_lc = np.percentile(lc_slice, [0.1, 98])

            imgs = [lc_slice, p_slice, gt_slice]
            titles = ["Low Count Input", f"U-Net Prediction ({point_folder})", "Ground Truth Target"]
            
            # Wir nutzen für LC eigene Lims, für Pred und GT aber die SHARED Lims
            v_params = [(vmin_lc, vmax_lc), (vmin_shared, vmax_shared), (vmin_shared, vmax_shared)]

            fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=100)

            for i, (ax, img, title, lims) in enumerate(zip(axes, imgs, titles, v_params)):
                vmin, vmax = lims
                if vmax <= vmin: vmax = vmin + 1e-5
                
                # Jetzt nutzen alle Bilder denselben Schwarz/Weiß Punkt
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nRange: {img.min():.2f} - {img.max():.2f}", fontsize=11, pad=12)
                ax.axis('off')

            plt.suptitle(f"Modell: {model_name} | Slice {idx:02d}", 
                         fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
            
            save_path = series_out / f"slice_{idx:02d}.png"
            plt.savefig(save_path)
            plt.close()

    print(f"\n✅ Fertig! Alle Bilder liegen in: {OUT_ROOT}")

if __name__ == "__main__":
    plot_selected_slices()