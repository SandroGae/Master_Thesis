import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. PARAMETER
# =====================================================
NPZ_FILE_NAME = "25D_replication_triple_loss_V2.npz"
NPZ_PATH = rf"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_test\25D_replication\{NPZ_FILE_NAME}"
OUTPUT_DIR = os.path.dirname(NPZ_PATH)
OUTPUT_PNG = os.path.join(OUTPUT_DIR, f"{NPZ_FILE_NAME}.png")

SLICES_TO_PLOT = [5, 20, 35] 

# =====================================================
# 2. DATEN LADEN
# =====================================================
if not os.path.exists(NPZ_PATH):
    raise FileNotFoundError(f"Die Datei {NPZ_PATH} existiert nicht.")

print(f"Lade Daten aus {NPZ_PATH}...")
data = np.load(NPZ_PATH)

low = data['low']
high = data['high']
pred = data['pred']

print(f"Shape der Arrays: Low={low.shape}, Pred={pred.shape}, High={high.shape}")

# =====================================================
# 3. KONTRAST BERECHNEN (DER FIX)
# =====================================================
# Wir nehmen die Ground Truth als Maßstab für den korrekten Kontrast
# Das 1% und 99% Perzentil ignoriert kaputte Pixel und spannt den perfekten Kontrast auf
vmin = np.percentile(high, 1)
vmax = np.percentile(high, 99)

print(f"Dynamischer Kontrast: vmin={vmin:.4f}, vmax={vmax:.4f}")

# =====================================================
# 4. PLOTTEN
# =====================================================
num_rows = len(SLICES_TO_PLOT)
fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))

for row_idx, slice_idx in enumerate(SLICES_TO_PLOT):
    ax_row = axes[row_idx] if num_rows > 1 else axes
    
    # --- LOW COUNT (Links) ---
    im0 = ax_row[0].imshow(low[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax_row[0].set_title(f"Low Count (Input) - Slice {slice_idx}", fontsize=14)
    ax_row[0].axis('off')

    # --- PREDICTION (Mitte) ---
    im1 = ax_row[1].imshow(pred[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax_row[1].set_title(f"Prediction - Slice {slice_idx}", fontsize=14)
    ax_row[1].axis('off')

    # --- GROUND TRUTH (Rechts) ---
    im2 = ax_row[2].imshow(high[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
    ax_row[2].set_title(f"High Count (GT) - Slice {slice_idx}", fontsize=14)
    ax_row[2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"Visualisierung erfolgreich gespeichert unter: {OUTPUT_PNG}")

plt.show()