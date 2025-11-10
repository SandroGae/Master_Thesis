#!/usr/bin/env python3
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

# ---------- Pfad einstellen ----------
FILE_TRAIN = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data\training_data.hdf5")

def msssim_with_per_scale(x, y,
                          max_val=1.0,
                          num_scales=5,
                          power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01, k2=0.03):
    """
    Liefert (per_scale_ssim, approx_msssim).
    per_scale_ssim: Tensor der Form (B, num_scales)
    approx_msssim:  Tensor der Form (B,)
    """
    assert len(power_factors) == num_scales
    xs, ys = x, y
    per_scale = []

    for s in range(num_scales):
        # SSIM pro Skala (Batch-Vektor)
        ssim_s = tf.image.ssim(xs, ys, max_val=max_val,
                               filter_size=filter_size,
                               filter_sigma=filter_sigma,
                               k1=k1, k2=k2)
        per_scale.append(ssim_s)

        # Für die nächste Skala: 2x2 Average-Pooling (Downsampling)
        if s < num_scales - 1:
            xs = tf.nn.avg_pool2d(xs, ksize=2, strides=2, padding="VALID")
            ys = tf.nn.avg_pool2d(ys, ksize=2, strides=2, padding="VALID")

            # Abbruch, falls Bilder kleiner als Filter werden
            h, w = xs.shape[1], xs.shape[2]
            if h is not None and w is not None and (h < filter_size or w < filter_size):
                # Restliche Skalen nicht mehr aussagekräftig
                # Auffüllen mit letzten verfügbaren SSIMs
                for _ in range(s+1, num_scales):
                    per_scale.append(ssim_s)
                break

    # (num_scales, B) -> (B, num_scales)
    per_scale = tf.stack(per_scale, axis=0)
    per_scale = tf.transpose(per_scale, [1, 0])

    # Gewichtetes Produkt über Skalen (Approx. zur TF-Implementierung)
    w = tf.constant(power_factors, dtype=per_scale.dtype)[None, :]  # (1,S)
    approx_msssim = tf.exp(tf.reduce_sum(w * tf.math.log(tf.clip_by_value(per_scale, 1e-6, 1.0)), axis=1))
    approx_msssim = tf.clip_by_value(approx_msssim, 0.0, 1.0)
    return per_scale, approx_msssim

# ---------- Hilfsfunktionen ----------
def to_nhwc(x_hwN):
    # (H, W, N) -> (N, H, W, 1)
    x = np.moveaxis(x_hwN, -1, 0)[..., None]
    return x.astype(np.float32)

def preprocess_like_val(x):
    """
    Repliziert deine Validation-Normierung:
      1) ReLU (Clipping auf [0, ∞))
      2) Summen-Normierung pro Bild (H,W,C)
      3) feste Skalierung 10000.0
      4) Clip auf [0,1]
    Erwartet x als (N,H,W,1), float32.
    """
    x = tf.nn.relu(x)
    sum_x = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) + 1e-12
    x = x / sum_x
    x = x * 10000.0
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x

# ---------- Daten laden ----------
with h5py.File(FILE_TRAIN, "r") as f:
    low_hwN  = f["low_count/data"][:]      # (H, W, N)
    high_hwN = f["high_count/data"][:]     # (H, W, N)

low  = to_nhwc(low_hwN)
high = to_nhwc(high_hwN)

# ---------- Erste 10 Paare auswählen ----------
start = 2000
end = 2010
low_10  = low[start:end]
high_10 = high[start:end]

# ---------- Auf TensorFlow-Tensoren und wie in val normalisieren ----------
low_tf  = tf.convert_to_tensor(low_10, dtype=tf.float32)
high_tf = tf.convert_to_tensor(high_10, dtype=tf.float32)

low_tf_p  = preprocess_like_val(low_tf)
high_tf_p = preprocess_like_val(high_tf)

# 1) Offizieller TF-Gesamtwert (Referenz)
tf_msssim = tf.image.ssim_multiscale(high_tf_p, low_tf_p, max_val=1.0)

# 2) Per-Skalen-SSIMs + approximierter MS-SSIM
per_scale_ssim, approx_msssim = msssim_with_per_scale(high_tf_p, low_tf_p, max_val=1.0)

# Ausgabe
ps = per_scale_ssim.numpy()        # (10, 5)
tf_m = tf_msssim.numpy()           # (10,)
ap_m = approx_msssim.numpy()       # (10,)

for i in range(ps.shape[0]):
    vals = "  ".join(f"S{s+1}:{ps[i,s]:.5f}" for s in range(ps.shape[1]))
    print(f"idx {i:02d} | {vals} | TF MS-SSIM:{tf_m[i]:.5f} | approx:{ap_m[i]:.5f}")

print("\nMittelwerte über 10 Bilder:")
print("Pro Skala:", "  ".join(f"S{s+1}:{ps[:,s].mean():.5f}" for s in range(ps.shape[1])))
print(f"TF MS-SSIM mean: {tf_m.mean():.5f}")
print(f"Approx     mean: {ap_m.mean():.5f}")
