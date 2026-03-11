import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# =====================================================
# SETUP
# =====================================================
BASE_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\test_files")
OUT_DIR  = BASE_DIR / "Diagnose_Plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def diagnose_and_plot(num_samples=5):
    all_files = sorted(list(BASE_DIR.glob("*.npz")))
    if not all_files:
        print(f"❌ Keine .npz Dateien gefunden.")
        return

    sampled_files = random.sample(all_files, min(num_samples, len(all_files)))

    for f_path in sampled_files:
        data = np.load(f_path)
        lc, pred, gt = data['lc'], data['pred'], data['gt']
        
        # --- SICHERHEITS-CHECK SHAPES ---
        if not (lc.shape == pred.shape == gt.shape):
            print(f"⚠️ WARNUNG: Shapes ungleich! LC:{lc.shape}, Pred:{pred.shape}, GT:{gt.shape}")
        
        idx = 20
        p_slice, gt_slice, lc_slice = pred[idx], gt[idx], lc[idx]

        # Plot Setup
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
        
        def get_lims(img):
            # Deine gewählten Perzentile für guten Kontrast
            return np.percentile(img, [0.5, 98])

        # Plotting
        v_params = [get_lims(lc_slice), get_lims(p_slice), get_lims(gt_slice)]
        imgs = [lc_slice, p_slice, gt_slice]
        titles = ["Low Count Input", "U-Net Prediction", "Ground Truth Target"]

        for i, (ax, img, title, lims) in enumerate(zip(axes, imgs, titles, v_params)):
            vmin, vmax = lims
            if vmax == vmin: vmax += 1e-5
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            # Titel mit Zeilenumbruch, um Überlappung zu vermeiden
            ax.set_title(f"{title}\nRange: {img.min():.2f} - {img.max():.2f}", fontsize=11, pad=12)
            ax.axis('off')

        # Supertitle fixen (y=0.98 und top-adjust)
        plt.suptitle(f"DanMAX Alignment Check: {f_path.stem} (Slice {idx})", 
                     fontsize=15, fontweight='bold', y=0.98)
        
        # FIX: tight_layout Platz für Titel lassen
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
        
        save_path = OUT_DIR / f"Diagnose_Fixed_{f_path.stem}.png"
        plt.savefig(save_path)
        print(f"➡️ Bild mit fixiertem Titel gespeichert: {save_path.name}")
        plt.close()

if __name__ == "__main__":
    diagnose_and_plot(5)