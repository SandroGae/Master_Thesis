# generate_movie_irunet.py
#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# Pfade & Konfiguration
# =====================================================
ROOT_DIR     = Path(r"C:\Users\sandr\VS_Master_Thesis")
DATA_DIR     = ROOT_DIR / "data" / "original_data"
WEIGHTS_PATH = ROOT_DIR / "JENS_IRUNET" / "JENS_IRUNET.hdf5"
H5_TEST_PATH = DATA_DIR / "test_data.hdf5"
MOVIES_DIR   = ROOT_DIR / "Plots" / "irunet_movies"

# Einstellungen
SERIES_IDX_1BASED = 12  # Welche Serie visualisiert werden soll (1..N)
FPS               = 3
SERIES_LEN        = 41  # Länge einer Temperatur-Serie

# ========= Projekt-Root in sys.path für jens_stuff =========
# Damit 'from jens_stuff import ...' funktioniert
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from jens_stuff import SumScaleNormalizer

# =====================================================
# 1. Modell-Architektur (IRUNet)
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
# 2. Daten laden & Normalisieren
# =====================================================
def load_series_raw(h5_path, series_idx_0based, series_len=41):
    """
    Lädt exakt die 41 Frames einer Serie als Rohdaten.
    Output: (41, H, W, 1)
    """
    start = series_idx_0based * series_len
    end   = start + series_len
    
    with h5py.File(h5_path, "r") as f:
        # HDF5 ist (H, W, N), wir slicen N
        low  = f["low_count/data"][:, :, start:end]
        high = f["high_count/data"][:, :, start:end]
        
    # Transpose zu (N, H, W, 1)
    low  = np.moveaxis(low, -1, 0)[..., None].astype(np.float32)
    high = np.moveaxis(high, -1, 0)[..., None].astype(np.float32)
    
    return low, high

def normalize_for_model(low_raw, high_raw):
    """
    Benutzt Jens' SumScaleNormalizer für das Modell-Input.
    """
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
    
    # Sicherheits-Clip und NaN Check
    Xn = tf.clip_by_value(tf.where(tf.math.is_finite(Xn), Xn, 0.), 0., 1.)
    Yn = tf.clip_by_value(tf.where(tf.math.is_finite(Yn), Yn, 0.), 0., 1.)
    
    return Xn.numpy(), Yn.numpy()

# =====================================================
# 3. Frame-Erzeugung (Visualisierung)
# =====================================================
def normalized_image(image):
    """
    Visuelle Normalisierung (Percentile Scaling) für den Plot.
    Macht das Bild 'hübsch' unabhängig von absoluten Werten.
    """
    vmin, vmax = np.percentile(image, [0.5, 99.5]) 
    if vmax - vmin < 1e-12:
        return image 
    return (image - vmin) / (vmax - vmin)

def create_frames_2d(X_seq, Y_pred, Y_true):
    """
    Erstellt Matplotlib-Figuren für jedes Bild in der Sequenz.
    Erwartet Input: (N, H, W, 1)
    """
    frames = []
    
    # Loop über die Zeitachse (N Frames)
    for i in range(X_seq.shape[0]):
        # Daten holen (H, W)
        inp_slice  = X_seq[i, :, :, 0]
        pred_slice = Y_pred[i, :, :, 0]
        gt_slice   = Y_true[i, :, :, 0]

        # Für Visualisierung skalieren (0-1 Bereich optimieren)
        inp_norm  = normalized_image(inp_slice)
        pred_norm = normalized_image(pred_slice)
        gt_norm   = normalized_image(gt_slice)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

        axes[0].imshow(inp_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Input (Low Count), Frame {i}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(pred_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"IRUNet Prediction, Frame {i}", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(gt_norm, cmap="gray_r", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Ground Truth (High Count), Frame {i}", fontsize=12)
        axes[2].axis("off")

        fig.tight_layout()
        
        # Bild in RGB-Array wandeln
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        
        plt.close(fig)
        frames.append(frame)

    return frames

# =====================================================
# Hauptfunktion
# =====================================================
def main():
    series_idx = SERIES_IDX_1BASED - 1  # 0-basiert
    
    out_name = f"IRUNet_Series_{SERIES_IDX_1BASED}.mp4"
    out_path = MOVIES_DIR / out_name

    print(f"--- IRUNet Video Generator ---")
    print(f"Gewichte:   {WEIGHTS_PATH}")
    print(f"Daten:      {H5_TEST_PATH}")
    print(f"Serie:      {SERIES_IDX_1BASED} (Index {series_idx})")
    print(f"Output:     {out_path}")

    # 1. Modell bauen & Laden
    print("Baue IRUNet und lade Gewichte...")
    model = build_irunet(input_shape=(192, 240, 1))
    model.load_weights(str(WEIGHTS_PATH))

    # 2. Daten laden
    print(f"Lade Rohdaten für Serie {SERIES_IDX_1BASED}...")
    low_raw, high_raw = load_series_raw(H5_TEST_PATH, series_idx, SERIES_LEN)
    print(f"Shape: {low_raw.shape}")

    # 3. Normalisieren (für das Modell)
    print("Normalisiere Daten (SumScale)...")
    X_norm, Y_norm = normalize_for_model(low_raw, high_raw)

    # 4. Prediction
    print("Berechne Prediction...")
    Y_pred = model.predict(X_norm, batch_size=4, verbose=1)

    # 5. Frames erstellen
    # Wir nutzen hier die normalisierten Daten für die Visualisierung, 
    # da `normalized_image` eh per Percentile skaliert. 
    # Das matcht das Verhalten deines Referenz-Codes.
    print("Erzeuge Video-Frames...")
    frames = create_frames_2d(X_norm, Y_pred, Y_norm)

    # 6. Video speichern
    MOVIES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Speichere Video mit {FPS} FPS...")
    imageio.mimsave(str(out_path), frames, fps=FPS)
    print("Fertig.")

if __name__ == "__main__":
    main()