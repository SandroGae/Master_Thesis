#!/usr/bin/env python3
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ==========================
# 1) Pfade & Auswahl
# ==========================
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet\Keras")

# Modelle, die im Ordner liegen
SELECT_LIST = [
    "VDSR_reconstruction_V2_loss0.0132_val0.0133_epochs100.keras",
    "unet_2d_simple_loss0.0135_val0.0142_epochs100.keras",  #loss: 0.014782 mae: 0.014782 mse: 0.000447 psnr_metric: 33.706829
    "unet_2d_simple_loss0.0118_val0.0135_epochs200.keras",  #loss: 0.014342 mae: 0.014342 mse: 0.000427 psnr_metric: 33.990610
    "unet_2d_SSIM_loss0.0605_val0.0787_epochs200.keras"     #loss: 0.137528 mae: 0.013313 mse: 0.000365 psnr_metric: 34.896782 ssim_metric: 0.861955
]

# ==========================
# 2) Hilfsfunktionen
# ==========================
CHUNK_N = 500   # Anzahl Bilder pro HDF5-Block
BATCH_SIZE = 32    # Keras Batch
FIX_SCALE_MIN = 10000.0
FIX_SCALE_MAX = 10001.0

def read_block(h5f, start, count):
    """Liest einen zusammenhaengenden N-Block und formt zu (M,H,W,1) um."""
    low_d  = h5f["low_count/data"]    # (H,W,N)
    high_d = h5f["high_count/data"]   # (H,W,N)
    end = min(start + count, low_d.shape[-1])
    sl = slice(start, end)            # N-Achse
    low  = np.moveaxis(np.asarray(low_d[...,  sl], dtype=np.float32), -1, 0)[..., None]
    high = np.moveaxis(np.asarray(high_d[..., sl], dtype=np.float32), -1, 0)[..., None]
    return low, high                  # (M,H,W,1)


def eval_hdf5_in_chunks(h5_path: Path, model, chunk_n=CHUNK_N, shuffle=False, seed=42, max_samples=None):
    sums = None
    total = 0
    with h5py.File(h5_path, "r") as f:
        Ntot = f["low_count/data"].shape[-1]

        # Auswahl begrenzen
        if max_samples is not None:
            Nsel = min(max_samples, Ntot)
        else:
            Nsel = Ntot

        order = np.arange(Ntot)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        # nur die ersten Nsel Indizes behalten
        order = order[:Nsel]

        for start in range(0, Nsel, chunk_n):
            block_idx = order[start:start+chunk_n]
            sort_idx = np.sort(block_idx)
            low, high = read_block(f, int(sort_idx[0]), len(sort_idx))
            inv = np.argsort(np.searchsorted(sort_idx, block_idx))
            low, high = low[inv], high[inv]
            M = low.shape[0]

            ds = (tf.data.Dataset.from_tensor_slices((low, high))
                .map(normalize_no_augment(10000.0, 10001.0), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))


            res = model.evaluate(ds, verbose=0, return_dict=True)
            if sums is None:
                sums = {k: float(v) * M for k, v in res.items()}
            else:
                for k, v in res.items():
                    sums[k] += float(v) * M
            total += M
            print(f"{total} done")

    avgs = {k: v / total for k, v in sums.items()}
    return avgs, total


def load_split(h5_path: Path):
    """lädt low/high aus deiner HDF5 und bringt sie in (N,H,W,1)"""
    with h5py.File(h5_path, "r") as f:
        low  = f["low_count/data"][:]   # (H,W,N)
        high = f["high_count/data"][:]  # (H,W,N)
    low  = np.moveaxis(low,  -1, 0)     # -> (N,H,W)
    high = np.moveaxis(high, -1, 0)

    low  = low[..., np.newaxis]         # -> (N,H,W,1)
    high = high[..., np.newaxis]
    return low.astype(np.float32), high.astype(np.float32)

def normalize_no_augment(scale_min: float, scale_max: float):
    """
    1) Clipping auf [0, ∞)
    2) Pro-Bild-Normierung über (H,W,C)
    3) Zufällige Skalierung in [scale_min, scale_max]
    4) Clipping auf [0, 1]
    Erwartet x,y als (H,W,C), float32.
    """
    def map_picture(x, y):
        # 1) nonnegativ
        x = tf.nn.relu(x)
        y = tf.nn.relu(y)

        # 2) L1-Normierung pro Bild
        sum_x = tf.reduce_sum(x, axis=[0, 1, 2], keepdims=True) + 1e-12
        sum_y = tf.reduce_sum(y, axis=[0, 1, 2], keepdims=True) + 1e-12
        x = x / sum_x
        y = y / sum_y

        # 3) zufällige Skalierung (pro Sample)
        scale = tf.random.uniform(shape=[1, 1, 1],
                                  minval=tf.cast(scale_min, tf.float32),
                                  maxval=tf.cast(scale_max, tf.float32),
                                  dtype=tf.float32)
        x = x * scale
        y = y * scale

        # 4) Clip auf [0,1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        y = tf.clip_by_value(y, 0.0, 1.0)
        return x, y
    return map_picture


# metriken wie im trainingscode
def psnr_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_mae_ssim(y_true, y_pred, alpha=0.7):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - alpha) * mae + alpha * (1.0 - ssim)

# ---- füge das oben hinzu ----
def ssim_loss(y_true, y_pred):
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def choose_model(list_):
    print("Verfügbare Modelle:")
    for i, name in enumerate(list_, start=1):
        print(f"  {i}) {name}")
    number = int(input("Choose your run (1–N): "))
    path = CHECKPOINT_DIR / list_[number - 1]
    print(f"\nLade Modell: {path}")

    # 1) Laden ohne Kompilierung (robust gegen Lambda-Loss)
    model = keras.models.load_model(
        str(path),
        custom_objects={
            # falls Layer/ops gebraucht werden
            "combined_mae_ssim": combined_mae_ssim,
            "psnr_metric": psnr_metric,
            "ssim_metric": ssim_metric,
            "<lambda>": ssim_loss,  # falls Checkpoint mit Lambda-Loss gespeichert wurde
        },
        compile=False,
    )

    # 2) IMMER mit den gewünschten Metriken kompilieren
    #    Heuristik für den Loss: SSIM-Modelle -> ssim_loss, sonst MAE (wie bei dir üblich)
    loss_fn = ssim_loss if "SSIM" in str(path.name) else "mae"
    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=["mae", "mse", psnr_metric, ssim_metric],
    )
    return model



# ==========================
# 3) main
# ==========================
if __name__ == "__main__":
    test_h5 = DATA_DIR / H5_NAME

    # Modell zuerst waehlen (damit Keras/TF initialisiert ist)
    model = choose_model(SELECT_LIST)

    # Gesamtdatensatz in 100er Bloecken evaluieren, optional geshuffelt
    avgs, total = eval_hdf5_in_chunks(
        h5_path=test_h5,
        model=model,
        chunk_n=CHUNK_N,
        shuffle=False,      # wenn du wirklich die "ersten" 1000 willst, nicht mischen
        max_samples=1000    # <- hier limitierst du
    )


    print(f"\n=== Gesamtauswertung ueber {total} Bilder (chunk={CHUNK_N}) ===")
    # avgs ist ein Dict: { 'loss': ..., 'mae': ..., ... }
    # Reihenfolge an metrics_names anlehnen, falls vorhanden:
    names = getattr(model, "metrics_names", list(avgs.keys()))
    for name in names:
        if name in avgs:
            print(f"{name:15s}: {avgs[name]:.6f}")

