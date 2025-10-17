# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
# JENS_IRUNET_MOVIE.py
from pathlib import Path
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import keras.layers as layers
import h5py

# === Aus deinem Code ===
from jens_stuff import SumScaleNormalizer   # gleiche Normalisierung wie im Training
# =======================

# ---------- IRUNet (2D) ----------
def build_irunet(input_shape=(192,240,1), n_filters=64, kernel_initializer="he_normal"):
    inp = tf.keras.Input(shape=input_shape)
    def inc(x,nf):
        a = layers.Conv2D(nf,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(x)
        c = layers.Conv2D(nf,3,padding="same",activation="relu",dilation_rate=2,kernel_initializer=kernel_initializer)(x)
        x2 = tf.keras.layers.Concatenate()([a,b,c])
        x2 = layers.Conv2D(nf,1,padding="same")(x2)
        return tf.keras.layers.Add()([x,x2])
    def inc_red(x, nf):
        sc = layers.Conv2D(nf, 2, strides=2, padding="same")(x)
        a  = layers.Conv2D(nf,   3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b  = layers.Conv2D(nf*2, 3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c  = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)
        x2 = tf.keras.layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return tf.keras.layers.Add()([sc, x2])

    h  = layers.Conv2D(n_filters,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(inp)
    c1 = inc_red(h,n_filters);  c1 = inc(c1,n_filters)
    c2 = inc_red(c1,n_filters); c2 = inc(c2,n_filters)
    c3 = inc_red(c2,n_filters); c3 = inc(c3,n_filters)
    b  = inc_red(c3,n_filters); b  = inc(b,n_filters)
    d3 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(b);  d3 = inc(d3,n_filters)
    d2 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d3); d2 = inc(d2,n_filters); d2 = tf.keras.layers.Add()([c2,d2])
    d1 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d2); d1 = inc(d1,n_filters); d1 = tf.keras.layers.Add()([c1,d1])
    t  = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d1); t  = inc(t,n_filters)
    out= layers.Conv2D(1,1,padding="same",activation="sigmoid")(t)
    return tf.keras.Model(inp,out,name="IRUNet")

# ---------- Daten laden (roh) ----------
def load_hwN(fp: Path):
    with h5py.File(fp,"r") as f:
        high = f["/high_count/data"][:].transpose(2,0,1)  # (N,H,W)
        low  = f["/low_count/data"] [:].transpose(2,0,1)
    return high, low

def load_splits(data_dir: Path):
    return {
        "train": load_hwN(data_dir/"training_data.hdf5"),
        "val":   load_hwN(data_dir/"validation_data.hdf5"),
        "test":  load_hwN(data_dir/"test_data.hdf5"),
    }

# ---------- Jens-Normalisierung (wie im Training) ----------
# Wir spiegeln die *Valid/Test*-Normalisierung aus deinem Training:
# SumScaleNormalizer(scale_range=[5000,5001], pre_offset=0, normalize_label=True,
#                    axis=(H,W,C), clip_before=[0,inf], clip_after=[0,1])
def normalize_with_jens(low_nhw, high_nhw):
    # Eingabe: (N,H,W) Counts
    X4 = low_nhw[..., None].astype(np.float32)   # (N,H,W,1)
    Y4 = high_nhw[..., None].astype(np.float32)  # (N,H,W,1)

    normalizer = SumScaleNormalizer(
        scale_range=[5000, 5001],
        pre_offset=0.0,
        normalize_label=True,
        axis=(1, 2, 3),      # wie im Val/Test-Setup beim Training
        batch_mode=True,     # GANZ WICHTIG: Batchmodus (B,H,W,C)
        clip_before=[0., float("inf")],
        clip_after=[0., 1.],
    )

    X_tf = tf.convert_to_tensor(X4, tf.float32)
    Y_tf = tf.convert_to_tensor(Y4, tf.float32)

    Xn, Yn = normalizer.map(X_tf, Y_tf)          # -> [0,1], gleiche Skala für X/Y
    # finite & clip wie in deinem map_slice_wise:
    Xn = tf.clip_by_value(tf.where(tf.math.is_finite(Xn), Xn, tf.zeros_like(Xn)), 0.0, 1.0)
    Yn = tf.clip_by_value(tf.where(tf.math.is_finite(Yn), Yn, tf.zeros_like(Yn)), 0.0, 1.0)

    return Xn.numpy(), Yn.numpy()                # (N,H,W,1)


# ---------- Auswahl: zentrale 37 Frames pro 41er Serie ----------
def select_center_frames_2_thru_38(X, Y, group_len=41):
    # X,Y: (N,H,W,1), N Vielfaches von 41
    N = X.shape[0]; assert N % group_len == 0, "N kein Vielfaches von 41"
    idx = np.concatenate([np.arange(g*group_len+2, g*group_len+39) for g in range(N//group_len)])
    return X[idx], Y[idx], (N//group_len)

# ---------- Anzeige-Helfer (identisch zu deinem Movie) ----------
def stretch_with_window(x01, lo, hi, gamma=0.8):
    x = np.clip(x01.astype(np.float32), 0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max() if x.max() > x.min() else 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    if gamma and gamma > 0: x = np.power(x, gamma)
    return (x*255.0 + 0.5).astype(np.uint8)

def make_rgb_labeled(panel_gray, label):
    h, w = panel_gray.shape; bar_h = 26
    canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    canvas.paste(Image.fromarray(panel_gray, "L").convert("RGB"), (0, bar_h))
    ImageDraw.Draw(canvas).text((6, 4), label, fill=(255,255,255), font=ImageFont.load_default())
    return np.asarray(canvas)

def hstack_same_height(imgs, pad_px=12):
    H = max(im.shape[0] for im in imgs)
    out_w = sum(int(round(im.shape[1]*H/im.shape[0])) for im in imgs) + pad_px*(len(imgs)-1)
    out = np.zeros((H, out_w, 3), np.uint8); x=0
    for i, im in enumerate(imgs):
        if im.shape[0] != H:
            new_w = int(round(im.shape[1]*H/im.shape[0]))
            im = np.asarray(Image.fromarray(im).resize((new_w, H), Image.BILINEAR))
        out[:, x:x+im.shape[1], :] = im
        x += im.shape[1] + (pad_px if i < len(imgs)-1 else 0)
    return out

def upscale_rgb(img_rgb, scale=2):
    if scale == 1: return img_rgb
    h, w = img_rgb.shape[:2]
    return np.asarray(Image.fromarray(img_rgb).resize((w*scale, h*scale), Image.BILINEAR))

def build_frames(low, pred, high, *, p_low=1.0, p_high=99.5, gamma=0.8, upscale=2, pad_px=12):
    frames=[]
    for i in range(low.shape[0]):
        num = float(high[i,...,0].sum())
        den = float(pred[i,...,0].sum())
        scale = num / max(den, 1e-8)
        pred_eq = np.clip(pred[i,...,0] * scale, 0.0, 1.0)

        lo, hi = np.percentile(high[i,...,0], [p_low, p_high])
        l8 = stretch_with_window(low [i,...,0], lo, hi, gamma)
        p8 = stretch_with_window(pred_eq       , lo, hi, gamma)
        h8 = stretch_with_window(high[i,...,0], lo, hi, gamma)

        l = make_rgb_labeled(l8,"Low-count")
        p = make_rgb_labeled(p8,"Denoised (Model)")
        h = make_rgb_labeled(h8,"High-count")
        frames.append(upscale_rgb(hstack_same_height([l,p,h], pad_px=pad_px), scale=upscale))
    return frames

def save_mp4(frames_rgb, out_path, fps=12):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=int(fps), quality=8) as w:
        for fr in frames_rgb: w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

# ---------- Main ----------
def main(model_weights: Path,
         data_dir: Path = Path.home()/ "data"/"original_data",
         out_dir:  Path = Path.home()/ "data"/"movies",
         fps: int = 12,
         spacer_seconds=0.25,
         p_low=1.0, p_high=99.5, gamma=0.8):

    # 1) Rohdaten laden
    splits = load_splits(data_dir)
    (high_train, low_train) = splits["train"]   # nur fuer evtl. Checks; Normalisierung ist per-frame (fixed scale)
    (high_test,  low_test ) = splits["test"]
    N, H, W = low_test.shape
    print(f"[INFO] Test N={N}, HxW={H}x{W} (41er-Serien)")

    # 2) Jens-Normalisierung *wie im Training* (val/test-Setup: fester Zielbereich 5000..5001)
    X_all, Y_all = normalize_with_jens(low_test, high_test)   # -> (N,H,W,1) in [0,1]

    # 3) Pro 41er Serie die zentralen 37 Frames: 2..38 (entspricht 3..39 1-basiert)
    X, Y, num_groups_total = select_center_frames_2_thru_38(X_all, Y_all, group_len=41)

    # 4) IRUNet bauen + Gewichte laden
    model = build_irunet(input_shape=(H, W, 1))
    model.load_weights(model_weights)

    # 5) Vorhersage
    pred = model.predict(X, batch_size=8, verbose=1)

    # 6) Frames bauen + Spacer zwischen Serien (genau wie im 3D-Movie)
    frames_all = []
    spacer_frames = max(1, int(round(spacer_seconds * fps)))
    # Dummy-Frame fuer Spacer
    tmp = build_frames(X[:1], pred[:1], Y[:1], p_low=p_low, p_high=p_high, gamma=gamma)[0]
    spacer = np.zeros_like(tmp)

    # gleiche Logik wie im 3D-Skript
    windows_per_group = 37
    use_groups = min(20, num_groups_total)  # nur die ersten 20

    for g in range(use_groups):
        s = g * windows_per_group
        e = s + windows_per_group
        frames_g = build_frames(X[s:e], pred[s:e], Y[s:e],
                                p_low=p_low, p_high=p_high, gamma=gamma)
        frames_all.extend(frames_g)
        if g < use_groups - 1:
            frames_all.extend([spacer] * spacer_frames)

    # 7) Speichern
    out_file = Path(out_dir) / f"{Path(model_weights).stem}_centers_3to39_IRUNet2D.mp4"
    save_mp4(frames_all, out_file, fps=fps)

if __name__ == "__main__":
    # Weights-only HDF5 von Jens’ 2D-IRUNet
    model_weights = Path("JENS_IRUNET.hdf5")
    main(model_weights)

