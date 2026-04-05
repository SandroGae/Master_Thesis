import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# SETUP
# =====================================================
# Lokaler Pfad zu den gezogenen .npz Dateien
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\Specialist_P14_40_models")
OUT_ROOT = BASE_DIR / "Rekonstruktionen_Test"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Wir wählen 5 repräsentative Slices aus der 40er-Serie aus
SELECTED_SLICES = [5, 12, 19, 26, 33]

# Da 5 Serien (0-4) vorliegen, wählen wir hier z.B. die erste (Index 0) zum Plotten
SERIES_IDX = 0 

def normalize_img(img):
    """Normiert ein Bild dynamisch auf den Bereich [0, 1] für maximalen Kontrast."""
    img_min, img_max = np.percentile(img, [0.1, 99.9])
    norm = (img - img_min) / (img_max - img_min + 1e-12)
    return np.clip(norm, 0, 1)

def plot_selected_slices():
    all_files = sorted(list(BASE_DIR.rglob("*.npz")))
    
    if not all_files:
        print(f"❌ Keine .npz Dateien gefunden in {BASE_DIR}")
        return

    print(f"🔍 Starte Visualisierung von {len(all_files)} Modellen...")

    for f_path in all_files:
        model_name = f_path.stem
        
        # Unterordner für jedes Modell anlegen
        model_out = OUT_ROOT / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        
        print(f"⌛ Plotting: {model_name}")
        
        # Daten laden (Shape von pred: (5, 40, 512, 512))
        data = np.load(f_path)
        pred = data['pred']
        
        # Prüfen, ob LC und GT vorhanden sind
        has_lc_gt = 'lc' in data and 'gt' in data
        if has_lc_gt:
            lc, gt = data['lc'], data['gt']

        for idx in SELECTED_SLICES:
            # Wir holen das Bild aus der gewählten Serie und dem gewählten Slice
            p_slice = pred[SERIES_IDX, idx]

            if has_lc_gt:
                gt_slice, lc_slice = gt[SERIES_IDX, idx], lc[SERIES_IDX, idx]

                # --- DER SKALIERUNGS-FIX ---
                lc_norm = normalize_img(lc_slice)
                p_norm = normalize_img(p_slice)
                gt_norm = normalize_img(gt_slice)

                imgs = [lc_norm, p_norm, gt_norm]
                titles = [
                    f"Low Count Input\n(Raw Range: {lc_slice.min():.2f} to {lc_slice.max():.2f})",
                    f"U-Net Prediction\n(Raw Range: {p_slice.min():.5f} to {p_slice.max():.5f})",
                    f"Ground Truth Target\n(Raw Range: {gt_slice.min():.2f} to {gt_slice.max():.2f})"
                ]

                fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=100)

                for ax, img, title in zip(axes, imgs, titles):
                    # vmin/vmax sind jetzt fest 0 und 1, da wir vorher normiert haben
                    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(title, fontsize=11, pad=12)
                    ax.axis('off')
            else:
                # Nur Prediction plotten
                p_norm = normalize_img(p_slice)

                fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
                ax.imshow(p_norm, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"U-Net Prediction\n(Raw Range: {p_slice.min():.5f} to {p_slice.max():.5f})", fontsize=11, pad=12)
                ax.axis('off')

            plt.suptitle(f"Modell: {model_name} | Serie {SERIES_IDX} | Slice {idx:02d}", 
                         fontsize=14, fontweight='bold', y=0.98 if has_lc_gt else 0.95)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.92] if has_lc_gt else [0, 0, 1, 0.9]) 
            
            save_path = model_out / f"slice_{idx:02d}.png"
            plt.savefig(save_path)
            plt.close()

    print(f"\n✅ Fertig! Alle Bilder liegen in: {OUT_ROOT}")

if __name__ == "__main__":
    plot_selected_slices()