import numpy as np
from pathlib import Path

# Pfad zu deiner NPZ
file_path = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\npz_files\Test_replication_V2\Eval_models_S01.npz")

def debug_npz(p):
    if not p.exists():
        print(f"❌ Datei nicht gefunden: {p}")
        return

    data = np.load(p)
    print(f"=== Analyse für: {p.name} ===")
    
    for key in data.files:
        arr = data[key]
        print(f"\n📊 Array: '{key}'")
        print(f"   Shape: {arr.shape}")
        print(f"   Dtype: {arr.dtype}")
        print(f"   Min:   {np.nanmin(arr):.6e}")
        print(f"   Max:   {np.nanmax(arr):.6e}")
        print(f"   Mean:  {np.nanmean(arr):.6e}")
        print(f"   Zeros: {np.count_nonzero(arr==0)} von {arr.size} Pixeln")
        print(f"   NaNs:  {np.isnan(arr).sum()}")
        print(f"   Infs:  {np.isinf(arr).sum()}")

        # Check Slice 20 spezifisch (weil wir das plotten)
        if len(arr.shape) == 3:
            slice_20 = arr[20]
            print(f"   Slice 20 Mean: {np.mean(slice_20):.6e}")

if __name__ == "__main__":
    debug_npz(file_path)
    