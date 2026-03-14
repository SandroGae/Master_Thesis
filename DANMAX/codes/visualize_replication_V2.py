import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# SETUP
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\Test_replication_V2")
OUT_ROOT = BASE_DIR / "Rekonstruktionen"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def plot_all_slices():
    all_files = sorted(list(BASE_DIR.glob("*.npz")))
    if not all_files:
        print(f"❌ Keine .npz Dateien in {BASE_DIR} gefunden.")
        return

    for f_path in all_files:
        print(f"⌛ Verarbeite: {f_path.name}...")
        data = np.load(f_path)
        lc, pred, gt = data['lc'], data['pred'], data['gt']

        # Ordner für die aktuelle Serie erstellen
        series_out = OUT_ROOT / f_path.stem
        series_out.mkdir(parents=True, exist_ok=True)

        # Slices 2 bis 37 rekonstruieren (insg. 36 Stück)
        for idx in range(2, 38):
            p_slice, gt_slice, lc_slice = pred[idx], gt[idx], lc[idx]

            # Plot Setup
            fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
            
            def get_lims(img, is_input=False):
                if is_input:
                    return np.percentile(img, [0.1, 95]) # Aggressiver Kontrast für verrauschten Input
                return np.percentile(img, [1, 99])       # Standard-Kontrast für Pred und GT

            imgs = [lc_slice, p_slice, gt_slice]
            titles = ["Low Count Input", "U-Net Prediction", "Ground Truth Target"]
            v_params = [get_lims(lc_slice, True), get_lims(p_slice), get_lims(gt_slice)]

            for i, (ax, img, title, lims) in enumerate(zip(axes, imgs, titles, v_params)):
                vmin, vmax = lims
                if vmax == vmin: vmax += 1e-5
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nRange: {img.min():.2f} - {img.max():.2f}", fontsize=11, pad=12)
                ax.axis('off')

            plt.suptitle(f"Rekonstruktion: {f_path.stem} | Slice {idx:02d}", 
                         fontsize=15, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
            
            # Speichern
            save_path = series_out / f"{f_path.stem}_slice_{idx:02d}.png"
            plt.savefig(save_path)
            plt.close()

        print(f"✅ Alle 36 Slices für {f_path.name} gespeichert in: {series_out.name}")

if __name__ == "__main__":
    plot_all_slices()