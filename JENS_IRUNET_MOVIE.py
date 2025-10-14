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
# #!/usr/bin/env python3
# JENS_IRUNET_MOVIE.py
# Lädt das IRUNet-Modell, lädt Gewichte aus .hdf5, erstellt Denoising-Videos.

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from unet_3d_data_JENS import prepare_in_memory_5to5


# ==========================================================
# 1) IRUNet-Definition direkt hier integriert
# ==========================================================
import keras.layers as layers

def build_irunet(input_shape=(192, 240, 1), n_filters=64, kernel_initializer="he_normal"):
    """Erstellt das IRUNet-Model wie in JENS_IRUNET.py"""
    inputs = tf.keras.Input(shape=input_shape)

    def inception_block(x, nf):
        a = layers.Conv2D(nf, 3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2, 3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c = layers.Conv2D(nf, 3, padding="same", activation="relu", dilation_rate=2, kernel_initializer=kernel_initializer)(x)
        concat = layers.Concatenate()([a, b, c])
        red = layers.Conv2D(nf, 1, padding="same")(concat)
        out = layers.Add()([x, red])
        return out

    def inception_block_reduction(x, nf):
        shortcut = layers.Conv2D(nf, 2, strides=2, padding="same")(x)
        a = layers.Conv2D(nf, 3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        b = layers.Conv2D(nf*2, 3, strides=2, padding="same", activation="relu", kernel_initializer=kernel_initializer)(x)
        c = layers.AveragePooling2D(padding="same")(x)
        concat = layers.Concatenate()([a, b, c])
        red = layers.Conv2D(nf, 1, padding="same")(concat)
        out = layers.Add()([shortcut, red])
        return out

    # Encoder
    head = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer=kernel_initializer)(inputs)
    conv1 = inception_block_reduction(head, n_filters)
    conv1 = inception_block(conv1, n_filters)
    conv2 = inception_block_reduction(conv1, n_filters)
    conv2 = inception_block(conv2, n_filters)
    conv3 = inception_block_reduction(conv2, n_filters)
    conv3 = inception_block(conv3, n_filters)
    body  = inception_block_reduction(conv3, n_filters)
    body  = inception_block(body, n_filters)

    # Decoder
    d3 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(body)
    d3 = inception_block(d3, n_filters)
    d2 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d3)
    d2 = inception_block(d2, n_filters)
    d2 = layers.Add()([conv2, d2])
    d1 = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d2)
    d1 = inception_block(d1, n_filters)
    d1 = layers.Add()([conv1, d1])
    tail = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same", activation="relu")(d1)
    tail = inception_block(tail, n_filters)
    tail = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(tail)

    return tf.keras.Model(inputs, tail, name="IRUNet")


# ==========================================================
# 2) Anzeige-Helfer (aus deinem alten Movie-Code)
# ==========================================================
def stretch_local_uint8(img01, p_low=1.0, p_high=99.5, gamma=0.8):
    x = np.clip(img01.astype(np.float32), 0.0, 1.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max() if x.max() > x.min() else 1.0)
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    if gamma is not None and gamma > 0:
        x = np.power(x, gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)

def make_rgb_labeled(panel_gray, label_text, pad=6):
    h, w = panel_gray.shape
    bar_h = 26
    canvas = Image.new("RGB", (w, h + bar_h + pad), (0, 0, 0))
    panel_img = Image.fromarray(panel_gray, mode="L").convert("RGB")
    canvas.paste(panel_img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), label_text, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(canvas)

def hstack_same_height(imgs, pad_px=8, pad_color=(0, 0, 0)):
    heights = [im.shape[0] for im in imgs]
    H = max(heights)
    resized = []
    for im in imgs:
        if im.shape[0] != H:
            scale = H / im.shape[0]
            new_w = int(round(im.shape[1] * scale))
            im = np.asarray(Image.fromarray(im).resize((new_w, H), Image.BILINEAR))
        resized.append(im)
    total_w = sum(im.shape[1] for im in resized) + pad_px * (len(resized) - 1)
    out = np.zeros((H, total_w, 3), dtype=np.uint8)
    out[...] = np.array(pad_color, dtype=np.uint8)
    x = 0
    for i, im in enumerate(resized):
        out[:, x:x+im.shape[1], :] = im
        x += im.shape[1]
        if i < len(resized) - 1:
            x += pad_px
    return out

def upscale_rgb(img_rgb, scale=2):
    if scale == 1:
        return img_rgb
    h, w = img_rgb.shape[:2]
    return np.asarray(Image.fromarray(img_rgb).resize((w*scale, h*scale), Image.BILINEAR))

def build_frames_2d(low, pred, high, p_low=1.0, p_high=99.5, gamma=0.8, layout="h", upscale=2, pad_px=12):
    assert low.shape == pred.shape == high.shape, "Shape mismatch!"
    frames = []
    for i in range(low.shape[0]):
        l = low[i, ..., 0]
        p = pred[i, ..., 0]
        h = high[i, ..., 0]
        l8 = stretch_local_uint8(l, p_low=p_low, p_high=p_high, gamma=gamma)
        p8 = stretch_local_uint8(p, p_low=p_low, p_high=p_high, gamma=gamma)
        h8 = stretch_local_uint8(h, p_low=p_low, p_high=p_high, gamma=gamma)
        l_rgb = make_rgb_labeled(l8, "Low-count")
        p_rgb = make_rgb_labeled(p8, "Denoised (IRUNet)")
        h_rgb = make_rgb_labeled(h8, "High-count")
        frame_rgb = hstack_same_height([l_rgb, p_rgb, h_rgb], pad_px=pad_px) if layout == "h" else np.concatenate([l_rgb, p_rgb, h_rgb], axis=0)
        frames.append(upscale_rgb(frame_rgb, scale=upscale))
    return frames

def save_mp4(frames_rgb, out_path, fps=12):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=fps, quality=8) as w:
        for fr in frames_rgb:
            w.append_data(fr)
    print(f"[OK] MP4 gespeichert: {out_path}")


# ==========================================================
# 3) Hauptfunktion
# ==========================================================
def main(model_weights: Path, data_dir: Path, out_dir: Path, fps=12):
    print(">>> Lade Testdaten...")
    (results, meta) = prepare_in_memory_5to5(
        data_dir=data_dir, size=1, group_len=1, percentile=99.9, dtype=np.float32
    )
    X_test, Y_test = results["test"]
    print(f"[INFO] Testshape: {X_test.shape}")

    print(f">>> Baue IRUNet und lade Gewichte aus {model_weights.name}")
    model = build_irunet(input_shape=(192, 240, 1))
    model.load_weights(model_weights)
    print("[OK] Gewichte geladen.")
    model.summary()

    print(">>> Vorhersage...")
    pred = model.predict(X_test, batch_size=8, verbose=1)

    print(">>> Frames bauen...")
    frames = build_frames_2d(X_test, pred, Y_test)
    out_file = out_dir / f"{Path(model_weights).stem}_movie.mp4"
    save_mp4(frames, out_file, fps=fps)


if __name__ == "__main__":
    model_weights = Path(__file__).parent / "JENS_IRUNET.hdf5"
    data_dir = Path.home() / "data" / "original_data"
    out_dir = Path.home() / "data" / "movies"

    main(model_weights, data_dir, out_dir, fps=12)

