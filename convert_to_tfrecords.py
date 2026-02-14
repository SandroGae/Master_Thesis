import os
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

# =====================================================
# 1. KONFIGURATION & PFADE
# =====================================================
DATA_DIR = Path.home() / "data" / "original_data"
# Wir speichern die TFRecords direkt in einen Unterordner von original_data
TFRECORD_DIR = DATA_DIR / "tfrecords"
TFRECORD_DIR.mkdir(parents=True, exist_ok=True)

SERIES_LEN = 10
DEPTH = 5

# =====================================================
# 2. HILFSFUNKTIONEN
# =====================================================
def load_split(h5_path):
    print(f"Lade {h5_path.name}...")
    with h5py.File(h5_path, "r") as f:
        lc = f["low_count/data"][:].astype('float32')
        hc = f["high_count/data"][:].astype('float32')
    
    # Achsen-Anpassung: (Slices, H, W, C)
    lc = np.moveaxis(lc, -1, 0)[:, :, :, np.newaxis]
    hc = np.moveaxis(hc, -1, 0)[:, :, :, np.newaxis]
    return lc, hc

def make_sliding_windows(X, y, series_len, depth):
    n_vols = series_len - depth + 1
    X_v, y_v = [], []
    for i in range(X.shape[0] // series_len):
        bx = X[i*series_len : (i+1)*series_len]
        by = y[i*series_len : (i+1)*series_len]
        for s in range(n_vols):
            X_v.append(bx[s:s+depth])
            y_v.append(by[s:s+depth])
    return np.stack(X_v), np.stack(y_v)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(name, X, y):
    out_path = TFRECORD_DIR / f"{name}.tfrecord"
    print(f"Schreibe {len(X)} Windows nach {out_path.name}...")
    
    with tf.io.TFRecordWriter(str(out_path)) as writer:
        for i in range(len(X)):
            # Konvertierung in binäre Features
            feature = {
                'X': _bytes_feature(X[i].tobytes()),
                'y': _bytes_feature(y[i].tobytes())
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"Erfolgreich erstellt: {out_path.name}")

# =====================================================
# 3. EXECUTION
# =====================================================
def main():
    files_to_process = [
        ("training", "training_data.hdf5"),
        ("validation", "validation_data.hdf5"),
        ("test", "test_data.hdf5")
    ]

    for name, filename in files_to_process:
        h5_path = DATA_DIR / filename
        if not h5_path.exists():
            print(f"Warnung: {filename} nicht gefunden. Überspringe...")
            continue
            
        X_raw, y_raw = load_split(h5_path)
        X_win, y_win = make_sliding_windows(X_raw, y_raw, SERIES_LEN, DEPTH)
        write_tfrecords(name, X_win, y_win)

    print("\n--- Alle Konvertierungen abgeschlossen! ---")

if __name__ == "__main__":
    main()