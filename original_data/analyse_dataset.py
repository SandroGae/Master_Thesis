import h5py
from pathlib import Path

def print_obj(name, obj):
    """Hilfsfunktion zum Rekursiven Drucken der Struktur"""
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  -> Dataset, shape = {obj.shape}, dtype = {obj.dtype}")
        # Optional: Nur anzeigen, wenn Chunks/Compression existieren
        if obj.chunks:
            print(f"     chunks = {obj.chunks}")
        if obj.compression:
            print(f"     compression = {obj.compression}")
    else:
        print("  -> Gruppe")

    if len(obj.attrs) > 0:
        print("  Attribute:")
        for k, v in obj.attrs.items():
            print(f"    {k}: {v}")
    print() # Leerzeile für bessere Lesbarkeit

def analyze_file(filepath):
    """Öffnet und analysiert eine einzelne HDF5 Datei"""
    print("=" * 60)
    print(f"ANALYSE DATEI: {filepath.name}")
    print("=" * 60)

    try:
        with h5py.File(filepath, "r") as f:
            print("\nFile-Attribute:")
            if len(f.attrs) > 0:
                for k, v in f.attrs.items():
                    print(f"  {k}: {v}")
            else:
                print("  (Keine globalen Attribute)")

            print(f"\nTop Level Keys: {list(f.keys())}")
            
            print("\nStruktur:")
            f.visititems(print_obj)
            
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden: {filepath}")
    except Exception as e:
        print(f"FEHLER beim Lesen der Datei: {e}")
    
    print("\n")

def main():
    base_dir = Path(__file__).resolve().parent
    
    # Liste der spezifischen Dateien, die analysiert werden sollen
    # (Die 'skip' Dateien werden hier bewusst weggelassen)
    target_files = [
        "training_data.hdf5",
        "validation_data.hdf5",
        "test_data.hdf5"
    ]

    for filename in target_files:
        full_path = base_dir / filename
        analyze_file(full_path)

if __name__ == "__main__":
    main()