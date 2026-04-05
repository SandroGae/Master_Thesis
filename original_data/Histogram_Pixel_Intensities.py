import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. SETUP
# =====================================================
DATA_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\original_data")
FILES = ["training_data.hdf5", "validation_data.hdf5", "test_data.hdf5"]
SCALE_FACTOR = 10000.0

def load_and_normalize(file_path):
    with h5py.File(file_path, "r") as f:
        lc = f["low_count/data"][:].astype("float32")
        hc = f["high_count/data"][:].astype("float32")
        
        lc = np.moveaxis(lc, -1, 0)
        hc = np.moveaxis(hc, -1, 0)
    
    # 1. ReLU (negative Werte zu 0)
    lc = np.maximum(lc, 0.0)
    hc = np.maximum(hc, 0.0)
    
    # 2. Sum-Normalization pro Slice + Scaling
    lc_sum = np.sum(lc, axis=(1, 2), keepdims=True) + 1e-12
    hc_sum = np.sum(hc, axis=(1, 2), keepdims=True) + 1e-12
    
    lc_norm = (lc / lc_sum) * SCALE_FACTOR
    hc_norm = (hc / hc_sum) * SCALE_FACTOR
    
    return lc_norm.flatten(), hc_norm.flatten()

def main():
    all_lc = []
    all_hc = []
    
    print("Lade und verarbeite Datensätze...")
    for file_name in FILES:
        file_path = DATA_DIR / file_name
        if file_path.exists():
            lc_flat, hc_flat = load_and_normalize(file_path)
            all_lc.append(lc_flat)
            all_hc.append(hc_flat)
            print(f"  - {file_name} geladen.")
        else:
            print(f"  - WARNUNG: {file_name} nicht gefunden!")
            
    if not all_lc:
        print("Keine Daten gefunden. Skript wird beendet.")
        return

    global_lc = np.concatenate(all_lc)
    global_hc = np.concatenate(all_hc)
    
    # =====================================================
    # 2. STATISTIKEN BERECHNEN UND AUSGEBEN
    # =====================================================
    print("\n" + "="*50)
    print("📊 GLOBALE STATISTIKEN (NACH NORMALISIERUNG)")
    print("="*50)
    
    print("LOW COUNT (Network Input):")
    print(f"  Maximum:       {np.max(global_lc):.4f}")
    print(f"  Pixel <= 1.0:  {np.mean(global_lc <= 1.0) * 100:.4f}%") # <--- NEU
    print(f"  Mean:          {np.mean(global_lc):.6f}")
    print(f"  Median:        {np.median(global_lc):.6f}")
    print("-" * 50)
    print("HIGH COUNT (Ground Truth / Network Target):")
    print(f"  Maximum:       {np.max(global_hc):.4f}")
    print(f"  Pixel <= 1.0:  {np.mean(global_hc <= 1.0) * 100:.4f}%") # <--- NEU
    print(f"  Mean:          {np.mean(global_hc):.6f}")
    print(f"  Median:        {np.median(global_hc):.6f}")
    print("="*50)

    # =====================================================
    # 3. PLOTTING (HISTOGRAMME)
    # =====================================================
    print("\nErstelle Histogramme (das kann einen Moment dauern)...")
    
    # Für logarithmische X-Achsen filtern wir exakte Nullen heraus (log(0) ist undefiniert)
    lc_pos = global_lc[global_lc > 0]
    hc_pos = global_hc[global_hc > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # Dynamische Bins basierend auf dem echten Maximum generieren
    max_val = max(np.max(global_lc), np.max(global_hc))
    log_bins = np.logspace(-4, np.log10(max_val + 10), 100)
    
    # --- PLOT 1: LOW COUNT ---
    ax1 = axes[0]
    ax1.hist(lc_pos, bins=log_bins, color='steelblue', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Sigmoid Clipping (1.0)')
    ax1.set_title('Low Count Intensities (Normalized Input)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pixel Intensity (Log Scale)', fontsize=12)
    ax1.set_ylabel('Pixel Count (Log Scale)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # --- PLOT 2: HIGH COUNT / GROUND TRUTH ---
    ax2 = axes[1]
    ax2.hist(hc_pos, bins=log_bins, color='seagreen', alpha=0.8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Sigmoid Clipping (1.0)')
    ax2.set_title('High Count Intensities (Ground Truth)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pixel Intensity (Log Scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    save_path = Path(r"C:\Users\sandr\VS_Master_Thesis\original_data")
    plt.savefig(save_path)
    print(f"\n>>> Plot erfolgreich gespeichert unter:\n{save_path}")

if __name__ == "__main__":
    main()