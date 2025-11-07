#!/usr/bin/env python3
from pathlib import Path
import h5py, numpy as np, tensorflow as tf
from tensorflow import keras

# ==========================
# 1) Pfade & Auswahl
# ==========================
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"            # enthält (H,W,N) low/high
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Keras")

SELECT_LIST = [
    "unet_3d_simple_loss0.0131_val0.0144_epochs100.keras",      # loss: 0.014804 mae: 0.014804 mse: 0.000452 psnr_metric: 33.691741 ssim_metric: 0.806831
    "unet_3d_simple_SSIM_loss0.0608_val0.0794_epochs100.keras"  # loss: 0.012999 mae: 0.012999 mse: 0.000352 psnr_metric: 35.024599 ssim_metric: 0.867231
]

# 3D Daten-Layout
SERIES_LEN = 41
DEPTH      = 5            # Volumen-Tiefe (D)
MID_SLICE  = 2            # mittlerer Slice (0-basiert): 0,1,2,3,4 -> 2

# Eval-Parameter
BATCH_SIZE = 8            # für CPU konservativer
FIX_SCALE  = 10000.0      # gleiche Skalierung wie Validation
MAX_VOLS   = 1000         # z.B. nur erste 1000 Volumes; None = alle

# ==========================
# 2) Normalisierung & Metriken (2D, weil nur mittlerer Slice)
# ==========================
def normalize_3d_per_slice_no_aug(scale: float = 10000.0):
    """
    Erwartet x,y: (D,H,W,C). Pro Slice L1-Norm, feste Skalierung, Clip [0,1], keine Augmentierung.
    """
    def map_vol(x, y):
        x = tf.nn.relu(x); y = tf.nn.relu(y)
        sum_x = tf.reduce_sum(x, axis=[1,2,3], keepdims=True) + 1e-12  # (D,1,1,1)
        sum_y = tf.reduce_sum(y, axis=[1,2,3], keepdims=True) + 1e-12
        x = tf.clip_by_value((x / sum_x) * scale, 0.0, 1.0)
        y = tf.clip_by_value((y / sum_y) * scale, 0.0, 1.0)
        return x, y
    return map_vol

def psnr_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Falls ein Checkpoint mit Lambda-Loss gespeichert wurde:
def ssim_loss(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# ==========================
# 3) Modell laden und auf "mittleren Slice ausgeben" wrappen
# ==========================
def choose_model(list_):
    print("Verfügbare 3D-Modelle:")
    for i, name in enumerate(list_, start=1):
        print(f"  {i}) {name}")
    number = int(input("Choose your run (1–N): "))
    path = CHECKPOINT_DIR / list_[number - 1]
    print(f"\nLade Modell: {path}")

    base = keras.models.load_model(
        str(path),
        custom_objects={
            "<lambda>": ssim_loss,  # falls Lambda-Loss drin ist
        },
        compile=False,
    )
    # Wrapper: gleiche Eingabe (5,H,W,1), Ausgabe = mittlerer Slice (H,W,1)
    inp = base.input
    mid = tf.keras.layers.Lambda(lambda t: t[:, MID_SLICE, ...], name="mid_slice")(base.output)
    model_mid = tf.keras.Model(inp, mid, name=base.name + "_midonly")

    # Metriken wie bei 2D:
    # Loss egal (wir optimieren ja nicht), nimm "mae" konsistent zu deinen 2D-Evals
    model_mid.compile(optimizer="adam",
                      loss="mae",
                      metrics=["mae", "mse", psnr_metric, ssim_metric])
    return model_mid

# ==========================
# 4) HDF5 -> Volumes streamen (pro Serie) und mittleren Slice als Label nehmen
# ==========================
def series_count(N, series_len=SERIES_LEN):
    assert N % series_len == 0, f"N={N} nicht durch series_len={series_len} teilbar"
    return N // series_len

def read_series_block(f, s_idx, series_len=SERIES_LEN):
    """Liest eine Serie von Länge 41 als (H,W,41) aus HDF5."""
    H, W, N = f["low_count/data"].shape
    start = s_idx * series_len
    sl = slice(start, start + series_len)
    low  = np.asarray(f["low_count/data"][...,  sl], dtype=np.float32)   # (H,W,41)
    high = np.asarray(f["high_count/data"][..., sl], dtype=np.float32)
    # (H,W,S) -> (S,H,W,1)
    low  = np.moveaxis(low,  -1, 0)[..., None]
    high = np.moveaxis(high, -1, 0)[..., None]
    return low, high  # (41,H,W,1)

def make_windows_from_series(low_ser, high_ser, depth=DEPTH):
    """Erzeugt Sliding Windows aus einer Serie: (41,H,W,1) -> (41-4, 5,H,W,1)."""
    S = low_ser.shape[0]
    n_vols = S - depth + 1
    X = np.empty((n_vols, depth, *low_ser.shape[1:]), dtype=np.float32)
    Y = np.empty_like(X)
    for i in range(n_vols):
        X[i] = low_ser[i:i+depth]
        Y[i] = high_ser[i:i+depth]
    return X, Y  # (n_vols,5,H,W,1)

def eval_3d_mid_in_chunks(h5_path: Path, model_mid, max_vols=None, batch_size=BATCH_SIZE):
    """
    Streamt pro Serie, baut Sliding-Window-Volumes (5,H,W,1),
    normalisiert pro Slice, bildet Label = mittlerer Slice und evaluiert.
    Gibt gewichtete Mittelwerte über alle Volumes zurück.
    """
    sums = None
    total = 0

    with h5py.File(h5_path, "r") as f:
        H, W, N = f["low_count/data"].shape
        n_ser   = series_count(N, SERIES_LEN)
        taken   = 0

        for s in range(n_ser):
            low_ser, high_ser = read_series_block(f, s, SERIES_LEN)      # (41,H,W,1)
            Xv, Yv = make_windows_from_series(low_ser, high_ser, DEPTH)  # (V,5,H,W,1)
            if max_vols is not None and taken + len(Xv) > max_vols:
                cutoff = max_vols - taken
                Xv = Xv[:cutoff]; Yv = Yv[:cutoff]

            # tf.data + Norm pro Slice + Label = mittlerer Slice
            ds = (tf.data.Dataset.from_tensor_slices((Xv, Yv))
                  .map(normalize_3d_per_slice_no_aug(FIX_SCALE), num_parallel_calls=tf.data.AUTOTUNE)
                  .map(lambda x,y: (x, y[MID_SLICE, ...]), num_parallel_calls=tf.data.AUTOTUNE)  # Label (H,W,1)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))

            res = model_mid.evaluate(ds, verbose=0, return_dict=True)
            M = Xv.shape[0]
            if sums is None:
                sums = {k: float(v) * M for k, v in res.items()}
            else:
                for k, v in res.items():
                    sums[k] += float(v) * M
            total += M
            taken += M
            print(f"{total} volumes done")

            if max_vols is not None and taken >= max_vols:
                break

    avgs = {k: v / total for k, v in sums.items()}
    return avgs, total

# ==========================
# 5) main
# ==========================
if __name__ == "__main__":
    test_h5 = DATA_DIR / H5_NAME
    model_mid = choose_model(SELECT_LIST)

    avgs, total = eval_3d_mid_in_chunks(
        h5_path=test_h5,
        model_mid=model_mid,
        max_vols=MAX_VOLS,           # None für alle Volumes
        batch_size=BATCH_SIZE,
    )

    print(f"\n=== 3D-MID-SLICE Gesamtauswertung über {total} Volumes ===")
    names = getattr(model_mid, "metrics_names", list(avgs.keys()))
    for name in names:
        if name in avgs:
            print(f"{name:15s}: {avgs[name]:.6f}")
