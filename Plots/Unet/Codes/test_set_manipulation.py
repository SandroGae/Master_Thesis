#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path


# konfiguration
IN_DIR = Path("/home/sgaell/data/original_data")
IN_FILE = IN_DIR / "test_data.hdf5"

OUT_DIR = Path("/home/sgaell/data/test_data_manipulated")
OUT_FILE = OUT_DIR / "test_every_second_image.hdf5"

SERIES_LEN_OLD = 41
SLICE_INDEX = 2 # Kick out every second image

def subsample_every_second(data_chunk, series_len):
    """
    Nimmt Daten im Format (H, W, N)
    Gruppiert in Serien der Länge series_len.
    Behält Index 0, 2, 4... (Jedes Zweite).
    Gibt (H, W, New_N) zurück.
    """
    H, W, N = data_chunk.shape
    
    assert N % series_len == 0, f"Datenanzahl {N} nicht durch {series_len} teilbar!"
    num_series = N // series_len

    data_t = np.transpose(data_chunk, (2, 0, 1))     # transponieren (H, W, N) -> (N, H, W)
    data_reshaped = data_t.reshape(num_series, series_len, H, W) # Reshapen in (Num_Series, Series_Len, H, W)

    # Slicing: Nimm jedes zweite Bild entlang Achse 1 (Series_Len)
    data_subsampled = data_reshaped[:, ::2, :, :] # Start bei 0, Schrittweite 2 (0, 2, 4, ..., 40)
    new_len = data_subsampled.shape[1]
    
    # Zurück flachen: (Num_Series * New_Len, H, W)
    new_N = num_series * new_len
    data_flat = data_subsampled.reshape(new_N, H, W)

    # Achsen zurücktauschen: (New_N, H, W) -> (H, W, New_N)
    data_final = np.transpose(data_flat, (1, 2, 0))

    print(f"  -> Neue Serienlänge: {new_len}")
    print(f"  -> Neuer Shape: {data_final.shape}")
    
    return data_final

def main():
    print(f"Öffne: {IN_FILE}")
    
    with h5py.File(IN_FILE, "r") as f_in:
        # Daten laden
        low_data = f_in["low_count/data"][:]
        high_data = f_in["high_count/data"][:]

    # Verarbeiten
    low_new = subsample_every_second(low_data, SERIES_LEN_OLD)
    high_new = subsample_every_second(high_data, SERIES_LEN_OLD)

    # Speichern
    print(f"\nSpeichere nach: {OUT_FILE}")

    with h5py.File(OUT_FILE, "w") as f_out:
        # Low Count Gruppe
        g_low = f_out.create_group("low_count")
        g_low.create_dataset("data", data=low_new, compression="gzip")
        
        # High Count Gruppe
        g_high = f_out.create_group("high_count")
        g_high.create_dataset("data", data=high_new, compression="gzip")

    print("Fertig.")

if __name__ == "__main__":
    main()