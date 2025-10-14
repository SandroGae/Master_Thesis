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
from unet_3d_data_JENS import prepare_in_memory_5to5

# ---------- IRUNet (2D) ----------
def build_irunet(input_shape=(192,240,1), n_filters=64, kernel_initializer="he_normal"):
    inp = tf.keras.Input(shape=input_shape)
    def inc(x,nf):
        a = layers.Conv2D(nf,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(x)
        c = layers.Conv2D(nf,3,padding="same",activation="relu",dilation_rate=2,kernel_initializer=kernel_initializer)(x)
        x2 = layers.Concatenate()([a,b,c])
        x2 = layers.Conv2D(nf,1,padding="same")(x2)
        return layers.Add()([x,x2])
    def inc_red(x, nf):
        sc = layers.Conv2D(nf, 2, strides=2, padding="same")(x)
        a  = layers.Conv2D(nf,   3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b  = layers.Conv2D(nf*2, 3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c  = layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)   # ← FIX
        x2 = layers.Concatenate()([a, b, c])
        x2 = layers.Conv2D(nf, 1, padding="same")(x2)
        return layers.Add()([sc, x2])


    h = layers.Conv2D(n_filters,3,padding="same",activation="relu",kernel_initializer=kernel_initializer)(inp)
    c1 = inc_red(h,n_filters);  c1 = inc(c1,n_filters)
    c2 = inc_red(c1,n_filters); c2 = inc(c2,n_filters)
    c3 = inc_red(c2,n_filters); c3 = inc(c3,n_filters)
    b  = inc_red(c3,n_filters); b  = inc(b,n_filters)
    d3 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(b);  d3 = inc(d3,n_filters)
    d2 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d3); d2 = inc(d2,n_filters); d2 = layers.Add()([c2,d2])
    d1 = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d2); d1 = inc(d1,n_filters); d1 = layers.Add()([c1,d1])
    t  = layers.Conv2DTranspose(n_filters,2,strides=2,padding="same",activation="relu")(d1); t  = inc(t,n_filters)
    out = layers.Conv2D(1,1,padding="same",activation="sigmoid")(t)
    return tf.keras.Model(inp,out,name="IRUNet")

# ---------- Daten laden (roh, ohne Fenster) ----------
def load_hwN(fp):
    with h5py.File(fp,"r") as f:
        high = f["/high_count/data"][:].transpose(2,0,1)  # (N,H,W)
        low  = f["/low_count/data"] [:].transpose(2,0,1)
    return high, low

def load_all_splits(data_dir: Path):
    data = {
        "train": load_hwN(data_dir/"training_data.hdf5"),
        "val":   load_hwN(data_dir/"validation_data.hdf5"),
        "test":  load_hwN(data_dir/"test_data.hdf5"),
    }
    return data

# ---------- Aus 41er-Serien die zentralen 37 ziehen (2..38) ----------
def select_center_range_2D(low_NHW, high_NHW, group_len=41, start=2, end=38):
    # low/high: (N,H,W), N vielfaches von 41
    N, H, W = low_NHW.shape
    assert N % group_len == 0, "N ist kein Vielfaches von group_len"
    idxs = []
    for g in range(N // group_len):
        base = g*group_len
        idxs.extend(range(base + start, base + end + 1))  # inkl. end
    idxs = np.array(idxs, dtype=np.int64)
    X = low_NHW [idxs][..., None].astype(np.float32)   # (B,H,W,1)
    Y = high_NHW[idxs][..., None].astype(np.float32)
    return X, Y

# ---------- Anzeige-Helfer (wie bei dir) ----------
def stretch_local_uint8(x, p_low=1.0, p_high=99.5, gamma=0.8):
    x = np.clip(x.astype(np.float32),0.0,1.0)
    lo, hi = np.percentile(x,[p_low,p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        lo, hi = float(x.min()), float(x.max() if x.max()>x.min() else 1.0)
    x = np.clip((x-lo)/(hi-lo+1e-8),0,1)
    if gamma and gamma>0: x = np.power(x,gamma)
    return (x*255.0+0.5).astype(np.uint8)

from PIL import Image, ImageDraw, ImageFont
def make_rgb_labeled(panel_gray, label):
    h,w = panel_gray.shape; bar_h=26
    canvas = Image.new("RGB",(w,h+bar_h),(0,0,0))
    canvas.paste(Image.fromarray(panel_gray,"L").convert("RGB"),(0,bar_h))
    d = ImageDraw.Draw(canvas); d.text((6,4),label,fill=(255,255,255),font=ImageFont.load_default())
    return np.asarray(canvas)
def hstack_same_height(imgs,pad_px=12):
    H = max(im.shape[0] for im in imgs); out_w = sum(int(round(im.shape[1]*H/im.shape[0])) for im in imgs) + pad_px*(len(imgs)-1)
    out = np.zeros((H,out_w,3),np.uint8); x=0
    for i,im in enumerate(imgs):
        if im.shape[0]!=H:
            new_w = int(round(im.shape[1]*H/im.shape[0])); im = np.asarray(Image.fromarray(im).resize((new_w,H), Image.BILINEAR))
        out[:,x:x+im.shape[1],:] = im; x += im.shape[1] + (pad_px if i<len(imgs)-1 else 0)
    return out
def upscale_rgb(img,scale=2):
    return np.asarray(Image.fromarray(img).resize((img.shape[1]*scale,img.shape[0]*scale), Image.BILINEAR)) if scale!=1 else img

def build_frames(low, pred, high, upscale=2):
    frames=[]
    for i in range(low.shape[0]):
        l8 = stretch_local_uint8(low [i,...,0]);  p8 = stretch_local_uint8(pred[i,...,0]);  h8 = stretch_local_uint8(high[i,...,0])
        l = make_rgb_labeled(l8,"Low-count"); p = make_rgb_labeled(p8,"Denoised (IRUNet)"); h = make_rgb_labeled(h8,"High-count")
        frames.append(upscale_rgb(hstack_same_height([l,p,h]),scale=upscale))
    return frames

def save_mp4(frames, out_path, fps=12):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=int(fps), quality=8) as w:
        for fr in frames: w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

# ---------- Main ----------
def main(model_weights: Path,
         data_dir: Path = Path.home()/ "data"/"original_data",
         out_dir:  Path = Path.home()/ "data"/"movies",
         fps:int=12, spacer_seconds=0.25):

    # 1) 5-zu-5 Datasets bauen (wie JENS es nutzt)
    results = prepare_in_memory_5to5(
        data_dir=data_dir,
        size=5,          # 5er Fenster
        group_len=41,
        dtype=np.float32
    )

    X5_test, Y5_test = results["test"]   # (B, 5, H, W, 1), Rohwerte (Counts) je nach Pipeline
    B, D, H, W, C = X5_test.shape
    assert D == 5 and C == 1, f"Unerwartete Shape: {X5_test.shape}"

    # 2) Mittleres Bild je 5er-Fenster (Index 2) extrahieren -> 2D-Frames fuer IRUNet
    X2d = X5_test[:, 2, ...]   # (B, H, W, 1)
    Y2d = Y5_test[:, 2, ...]   # (B, H, W, 1)

    # 3) Pro 41er Serie nur die zentralen 37 Fenster nehmen (Fensterstarts 2..38)
    windows_per_group = 41 - 5 + 1   # = 37
    num_groups_total  = B // windows_per_group
    idx = np.concatenate([np.arange(g*windows_per_group, g*windows_per_group + windows_per_group) 
                        for g in range(num_groups_total)])
    X_test = X2d[idx]   # (37*num_groups, H, W, 1)
    Y_test = Y2d[idx]

    # 4) Vorhersage
    pred = model.predict(X_test, batch_size=8, verbose=1)

    # 5) Frames bauen und Spacer zwischen Serien einfuegen
    frames_all = []
    windows_per_group = 37
    spacer_frames = max(1, int(round(spacer_seconds*fps)))
    # Dummy-frame fuer Spacer (schwarz)
    tmp = build_frames(X_test[:1], pred[:1], Y_test[:1])[0]
    spacer = np.zeros_like(tmp)

    total_groups = (N // 41)
    for g in range(total_groups):
        s = g*windows_per_group
        e = s+windows_per_group
        frames_g = build_frames(X_test[s:e], pred[s:e], Y_test[s:e])
        frames_all.extend(frames_g)
        if g < total_groups-1:
            frames_all.extend([spacer]*spacer_frames)

    # 6) Speichern
    out_file = Path(out_dir) / f"{Path(model_weights).stem}_centers_3to39.mp4"
    save_mp4(frames_all, out_file, fps=fps)

if __name__ == "__main__":
    model_weights = Path("JENS_IRUNET.hdf5")  # Weights-only HDF5
    main(model_weights)

