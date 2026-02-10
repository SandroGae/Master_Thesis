# inference_irunet.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import h5py

# =====================================================
# Konfiguration
# =====================================================
ROOT_DIR     = Path(r"C:\Users\sandr\VS_Master_Thesis")
WEIGHTS_PATH = ROOT_DIR / "JENS_IRUNET" / "JENS_IRUNET.hdf5"
H5_TEST_PATH = ROOT_DIR / "original_data" / "test_data.hdf5"
OUT_DIR      = ROOT_DIR / "Plots" / "Unet" / "Analysis_ROI" / "Predictions_Raw"

SERIES_IDX_1BASED = 12  # Wir nehmen Serie 12 für den Vergleich
SERIES_LEN        = 41
INPUT_SHAPE       = (192, 240, 1)

# =====================================================
# 1. Helper: SumScaleNormalizer (aus jens_stuff)
# =====================================================
class SumScaleNormalizer:
    def __init__(self, scale_range=[5000, 15000], pre_offset=0.0, normalize_label=True, 
                 axis=(1, 2, 3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0., 1.]):
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.axis = axis
        self.clip_before = clip_before
        self.clip_after = clip_after
        self.epsilon = 1e-12

    def _norm(self, img):
        # 1. Clip Before
        if self.clip_before:
            if self.clip_before[0] is not None:
                 img = tf.maximum(img, self.clip_before[0])
            if self.clip_before[1] != float("inf"):
                 img = tf.minimum(img, self.clip_before[1])

        # 2. Summe
        sums = tf.reduce_sum(img, axis=self.axis, keepdims=True)
        sums = tf.maximum(sums, self.epsilon)

        # 3. Scale (Fixed to min for inference)
        target_scale = self.scale_min 
        img_norm = (img / sums) * target_scale

        # 4. Clip After
        if self.clip_after:
            img_norm = tf.clip_by_value(img_norm, self.clip_after[0], self.clip_after[1])
        return img_norm

    def map(self, x, y=None):
        x = tf.cast(x, tf.float32)
        x_n = self._norm(x)
        if y is not None:
            y = tf.cast(y, tf.float32)
            y_n = self._norm(y)
            return x_n, y_n
        return x_n

# =====================================================
# 2. Modell-Architektur (IRUNet)
# =====================================================
def build_irunet(input_shape=(192,240,1), n_filters=64, kernel_initializer="he_normal"):
    inp = keras.Input(shape=input_shape)

    def inc(x, nf):
        a = layers.Conv2D(nf,   3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2, 3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c = layers.Conv2D(nf,   3, padding="same", activation="relu", dilation_rate=2, kernel_initializer=kernel_initializer)(x)
        x2 = layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return layers.Add()([x, x2])

    def inc_red(x, nf):
        sc = layers.Conv2D(nf, 2, strides=2, padding="same")(x)
        a  = layers.Conv2D(nf,   3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b  = layers.Conv2D(nf*2, 3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c  = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)
        x2 = layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return layers.Add()([sc, x2])

    h  = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(inp)
    c1 = inc_red(h, n_filters);  c1 = inc(c1, n_filters)
    c2 = inc_red(c1, n_filters); c2 = inc(c2, n_filters)
    c3 = inc_red(c2, n_filters); c3 = inc(c3, n_filters)
    b  = inc_red(c3, n_filters); b  = inc(b, n_filters)

    d3 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(b)
    d3 = inc(d3, n_filters)
    d2 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d3)
    d2 = inc(d2, n_filters)
    d2 = layers.Add()([c2, d2])
    d1 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d2)
    d1 = inc(d1, n_filters)
    d1 = layers.Add()([c1, d1])
    t  = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d1)
    t  = inc(t, n_filters)

    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(t)
    return keras.Model(inp, out, name="IRUNet")

# =====================================================
# 3. Daten laden & Normalisieren
# =====================================================
def load_series_raw(h5_path, series_idx_0based, series_len=41):
    start = series_idx_0based * series_len
    end   = start + series_len
    with h5py.File(h5_path, "r") as f:
        low  = f["low_count/data"][:, :, start:end]
        high = f["high_count/data"][:, :, start:end]
        
    low  = np.moveaxis(low, -1, 0)[..., None].astype(np.float32)   # (41, 192, 240, 1)
    high = np.moveaxis(high, -1, 0)[..., None].astype(np.float32)  # (41, 192, 240, 1)
    return low, high

def normalize_for_model(low_raw, high_raw):
    # Exakt die Einstellungen aus generate_movie_irunet.py
    normalizer = SumScaleNormalizer(
        scale_range=[5000, 5001],
        pre_offset=0.0,
        normalize_label=True,
        axis=(1,2,3),
        batch_mode=True,
        clip_before=[0., float("inf")],
        clip_after=[0., 1.],
    )
    Xn, Yn = normalizer.map(tf.convert_to_tensor(low_raw),
                            tf.convert_to_tensor(high_raw))
    
    # Zusätzlicher Clip wie im Originalcode
    Xn = tf.clip_by_value(tf.where(tf.math.is_finite(Xn), Xn, 0.), 0., 1.)
    Yn = tf.clip_by_value(tf.where(tf.math.is_finite(Yn), Yn, 0.), 0., 1.)
    
    return Xn.numpy(), Yn.numpy()

# =====================================================
# Main
# =====================================================
def main():
    print(f"--- IRUNet Inference ---")
    print(f"Gewichte: {WEIGHTS_PATH}")
    
    # 1. Modell bauen & Laden
    model = build_irunet(input_shape=INPUT_SHAPE)
    model.load_weights(str(WEIGHTS_PATH))
    print("Modell geladen.")

    # 2. Daten laden (Serie 12)
    series_idx = SERIES_IDX_1BASED - 1
    print(f"Lade Serie {SERIES_IDX_1BASED} (Index {series_idx})...")
    low_raw, high_raw = load_series_raw(H5_TEST_PATH, series_idx, SERIES_LEN)
    
    # 3. Normalisieren (für Input ins Netz)
    print("Normalisiere...")
    X_norm, Y_norm = normalize_for_model(low_raw, high_raw)

    # 4. Prediction
    print("Berechne Prediction...")
    # Da IRUNet 2D ist und Batch-Size Input akzeptiert, können wir einfach das ganze Array (41, 192, 240, 1) reinwerfen
    Y_pred = model.predict(X_norm, batch_size=4, verbose=1)

    # 5. Daten aufbereiten für NPZ (Squeeze channels)
    # Shape wird (41, 192, 240)
    lc_out   = np.squeeze(X_norm, axis=-1)
    pred_out = np.squeeze(Y_pred, axis=-1)
    gt_out   = np.squeeze(Y_norm, axis=-1)

    # 6. Speichern
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / f"Pred_IRUNet_Series_{SERIES_IDX_1BASED}.npz"
    
    print(f"Speichere Ergebnisse: {outfile}")
    np.savez_compressed(outfile, lc=lc_out, pred=pred_out, gt=gt_out)
    print("Fertig.")

if __name__ == "__main__":
    main()