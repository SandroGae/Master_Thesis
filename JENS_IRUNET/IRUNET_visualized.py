# IRUNet_infer_rawviz.py
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# ========= Pfade anpassen =========
DATA_DIR   = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
WEIGHTS    = Path(r"C:\Users\sandr\VS_Master_Thesis\JENS_IRUNET\JENS_IRUNET.hdf5")
OUT_DIR    = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\irunet_rawviz")

# SELECTED_SAMPLES = [1,2,3,4,5]   # beliebige Indizes zum visualisieren
SUM_EQUALIZE_TO_LABEL = True   # Prediction auf Label-Summe skalieren (für faire Roh-Vergleiche)
SAVE_DPI   = 200

# ========= Projekt-Root in sys.path, damit jens_stuff importierbar ist =========
ROOT = Path(__file__).resolve().parents[1]  # ...\VS_Master_Thesis
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---- Normalizer wie bei Jens ----
from jens_stuff import SumScaleNormalizer  # liegt bei dir im Projektroot

# ========= IRUNet (2D) exakt wie in deiner Vorlage =========
def build_irunet(input_shape=(192,240,1), n_filters=64, kernel_initializer="he_normal"):
    inp = keras.Input(shape=input_shape)

    def inc(x, nf):
        a = layers.Conv2D(nf,   3, padding="same", activation="relu",
                          kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2, 3, padding="same", activation="relu",
                          kernel_initializer=kernel_initializer)(x)
        c = layers.Conv2D(nf,   3, padding="same", activation="relu",
                          dilation_rate=2, kernel_initializer=kernel_initializer)(x)
        x2 = layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return layers.Add()([x, x2])

    def inc_red(x, nf):
        sc = layers.Conv2D(nf, 2, strides=2, padding="same")(x)
        a  = layers.Conv2D(nf,   3, strides=2, padding="same", activation="relu",
                           kernel_initializer=kernel_initializer)(x)
        b  = layers.Conv2D(nf*2, 3, strides=2, padding="same", activation="relu",
                           kernel_initializer=kernel_initializer)(x)
        c  = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)
        x2 = layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return layers.Add()([sc, x2])

    h  = layers.Conv2D(n_filters, 3, padding="same", activation="relu",
                       kernel_initializer=kernel_initializer)(inp)
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

# ========= Daten laden (Rohwerte, keine Vorverarbeitung) =========
def load_hwN(fp: Path):
    with h5py.File(fp, "r") as f:
        high = f["/high_count/data"][:]  # (H,W,N)
        low  = f["/low_count/data"][:]   # (H,W,N)
    high = high.transpose(2,0,1)[..., None].astype(np.float32)  # (N,H,W,1)
    low  = low.transpose(2,0,1)[..., None].astype(np.float32)
    return high, low

# ========= Normalisieren wie bei Jens; danach inverse auf Prediction =========
def normalize_like_jens(low_raw, high_raw):
    normalizer = SumScaleNormalizer(
        scale_range=[5000, 5001],   # fester Bereich (wie bei Val/Test)
        pre_offset=0.0,
        normalize_label=True,       # X und y gleich skaliert
        axis=(1,2,3),               # über (H,W,C) je Sample
        batch_mode=True,
        clip_before=[0., float("inf")],
        clip_after=[0., 1.],
    )
    Xn, Yn = normalizer.map(tf.convert_to_tensor(low_raw),
                            tf.convert_to_tensor(high_raw))
    # NaNs/Infs abfangen + in [0,1] clippen – reine Sicherheit
    Xn = tf.clip_by_value(tf.where(tf.math.is_finite(Xn), Xn, 0.), 0., 1.)
    Yn = tf.clip_by_value(tf.where(tf.math.is_finite(Yn), Yn, 0.), 0., 1.)
    return Xn.numpy(), Yn.numpy(), normalizer

# ========= robuste Visualisierungsgrenzen je Panel =========
def robust_minmax(img, p1=1, p99=99):
    vals = img[...,0].ravel()
    vmin, vmax = np.percentile(vals, (p1, p99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals) + 1e-6)
    return float(vmin), float(vmax)

def show_triplet(low_raw, high_raw, pred_raw, save_path=None, title=None, cmap="gray_r"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = [
        ("Low Count",  low_raw),
        ("High Count", high_raw),
        ("IRUNet (pred, denorm)", pred_raw),
    ]
    for ax, (ttl, img) in zip(axes, panels):
        vmin, vmax = robust_minmax(img, 1, 99)
        ax.imshow(img[...,0], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(ttl)
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def main():
    # 1) Rohdaten laden
    high_raw_all, low_raw_all = load_hwN(DATA_DIR / "test_data.hdf5")  # (N,H,W,1)

    # 2) Indizes sauber wählen (z.B. jedes 41. Sample ab Index 20)
    idxs = list(range(20, high_raw_all.shape[0], 41))
    if not idxs:
        raise RuntimeError("Keine passenden Indizes gefunden (prüfe N und Start/Schritt).")

    # 3) Batch zusammenstellen
    high_raw = high_raw_all[idxs]   # (B,H,W,1)
    low_raw  = low_raw_all[idxs]    # (B,H,W,1)

    # 4) Normalisieren wie bei Jens (nur fürs Modell)
    Xn, Yn, normalizer = normalize_like_jens(low_raw, high_raw)

    # 5) Modell bauen & Gewichte laden
    H, W = Xn.shape[1], Xn.shape[2]
    model = build_irunet(input_shape=(H, W, 1))
    model.load_weights(str(WEIGHTS))

    # 6) Vorhersage
    pred_norm = model(Xn, training=False).numpy()

    # 7) Inverse Normalisierung der Prediction
    pred_tensor = tf.convert_to_tensor(pred_norm)
    try:
        pred_raw = normalizer.inverse_map(pred_tensor).numpy()
    except TypeError:
        pred_raw = normalizer.inverse_map(pred_tensor, length=3).numpy()

    # 8) Optional: Prediction auf Label-Summe skalieren (fairer Rohvergleich)
    if SUM_EQUALIZE_TO_LABEL:
        s_pred = np.sum(np.clip(pred_raw, 0, None), axis=(1,2,3), keepdims=True)
        s_lab  = np.sum(np.clip(high_raw, 0, None), axis=(1,2,3), keepdims=True)
        scale  = np.divide(s_lab, s_pred, out=np.ones_like(s_lab), where=(s_pred > 0))
        pred_raw = pred_raw * scale

    # 9) Visualisieren & speichern – ueber die Sample-Achse iterieren
    for i in range(low_raw.shape[0]):
        save_path = OUT_DIR / f"test_triplet_{i:03d}.png"
        show_triplet(low_raw[i], high_raw[i], pred_raw[i],
                     save_path=save_path, title=f"Sample {idxs[i]}")

    print(f"Fertig. {low_raw.shape[0]} Abbildungen gespeichert in: {OUT_DIR}")


if __name__ == "__main__":
    main()
