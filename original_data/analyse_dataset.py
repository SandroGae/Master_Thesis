import h5py
from pathlib import Path

def print_obj(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  -> Dataset, shape = {obj.shape}, dtype = {obj.dtype}")
        print(f"     chunks = {obj.chunks}, compression = {obj.compression}")
    else:
        print("  -> Gruppe")

    if len(obj.attrs) > 0:
        print("  Attribute:")
        for k, v in obj.attrs.items():
            print(f"    {k}: {v}")
    print()

def main():
    base_dir = Path(__file__).resolve().parent
    fname = base_dir / "training_data.hdf5"

    with h5py.File(fname, "r") as f:
        print(f"Datei: {fname}")
        print("\nFile-Attribute:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")

        print("\nTop Level Keys:", list(f.keys()))
        print("\nStruktur:")
        f.visititems(print_obj)

if __name__ == "__main__":
    main()

