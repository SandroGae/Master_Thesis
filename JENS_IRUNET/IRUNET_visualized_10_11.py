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
    GROUP_LEN = 41
    # Das waren zuvor die Positionen in deiner 20/41-Auswahl:
    CHOSEN_POS_IN_SELECTION = [10, 11]   # entspricht test_triplet_010 und _011
    START_IN_SERIES = 20                 # deine alte 20/41-Logik

    # 1) Daten laden
    high_all, low_all = load_hwN(DATA_DIR / "test_data.hdf5")  # (N,H,W,1)
    N = high_all.shape[0]

    # 2) Absolutindizes der beiden Frames, die du vorher gerendert hast
    abs_indices = [START_IN_SERIES + p * GROUP_LEN for p in CHOSEN_POS_IN_SELECTION]

    # 3) Startindex der zugehörigen Serien bestimmen (0-basiert, je 41 Frames)
    series_starts = sorted({(idx // GROUP_LEN) * GROUP_LEN for idx in abs_indices})

    # 4) Alle 41 Indizes beider Serien einsammeln
    idxs = []
    meta = []  # (series_id_0based, frame_in_series_0based)
    for s_start in series_starts:
        s_end = s_start + GROUP_LEN
        if s_end > N:
            raise RuntimeError(f"Serie ab {s_start} geht über N={N} hinaus.")
        series_id = s_start // GROUP_LEN  # 0-basiert
        for k in range(GROUP_LEN):
            idxs.append(s_start + k)
            meta.append((series_id, k))

    # 5) Batch bauen
    high_raw = high_all[idxs]
    low_raw  = low_all[idxs]

    # 6) Normalisieren -> wie bei Jens
    Xn, Yn, normalizer = normalize_like_jens(low_raw, high_raw)

    # 7) Modell + Gewichte
    H, W = Xn.shape[1], Xn.shape[2]
    model = build_irunet(input_shape=(H, W, 1))
    model.load_weights(str(WEIGHTS))

    # 8) Inferenz
    pred_norm = model(Xn, training=False).numpy()

    # 9) Denormalisieren
    pred_tensor = tf.convert_to_tensor(pred_norm)
    try:
        pred_raw = normalizer.inverse_map(pred_tensor).numpy()
    except TypeError:
        pred_raw = normalizer.inverse_map(pred_tensor, length=3).numpy()

    # 10) Optional: Summen angleichen
    if SUM_EQUALIZE_TO_LABEL:
        s_pred = np.sum(np.clip(pred_raw, 0, None), axis=(1,2,3), keepdims=True)
        s_lab  = np.sum(np.clip(high_raw, 0, None), axis=(1,2,3), keepdims=True)
        scale  = np.divide(s_lab, s_pred, out=np.ones_like(s_lab), where=(s_pred > 0))
        pred_raw = pred_raw * scale

    # 11) Speichern mit neuen Namen: serie_{ID}_frame_{00..40}.png
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(low_raw.shape[0]):
        series_id, frame_idx = meta[i]  # beide 0-basiert
        save_path = OUT_DIR / f"serie_{series_id:03d}_frame_{frame_idx:02d}.png"
        title = f"Serie {series_id:03d} – Frame {frame_idx:02d}"
        show_triplet(low_raw[i], high_raw[i], pred_raw[i], save_path=save_path, title=title)

    print(f"Fertig. {low_raw.shape[0]} Abbildungen (2×41) gespeichert in: {OUT_DIR}")


if __name__ == "__main__":
    main()