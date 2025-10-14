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
# Baut ein MP4-Video mit Low | Denoised (IRUNet) | High für 2D-Daten.
# Nutzt dasselbe Layout- und Stretching-System wie eval_make_movie_unet3d.py.

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from unet_3d_data import prepare_in_memory  # wird auch für 2D-Slices verwendet

# ---------- Anzeige-Helfer ----------
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
    font = ImageFont.load_default()
    draw.text((6, 4), label_text, fill=(255, 255, 255), font=font)
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

# ---------- Frames bauen ----------
def build_frames_2d(low, pred, high, *, p_low=1.0, p_high=99.5, gamma=0.8, layout="h", upscale=2, pad_px=12):
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

        if layout == "h":
            frame_rgb = hstack_same_height([l_rgb, p_rgb, h_rgb], pad_px=pad_px)
        else:
            frame_rgb = np.concatenate([l_rgb, p_rgb, h_rgb], axis=0)

        frame_rgb = upscale_rgb(frame_rgb, scale=upscale)
        frames.append(frame_rgb)
    return frames

# ---------- Speichern ----------
def save_mp4(frames_rgb, out_path, fps=12):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=fps, quality=8) as w:
        for fr in frames_rgb:
            w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

# ---------- Main ----------
def main(
    model_path: Path,
    data_dir: Path = Path.home() / "data" / "original_data",
    out_dir: Path = Path.home() / "data" / "movies",
    fps: int = 12,
    p_low=1.0, p_high=99.5, gamma=0.8,
    upscale=2
):
    print(">>> Lade Testdaten (ALT-Pipeline)...")
    (results, meta) = prepare_in_memory(
        data_dir=data_dir,
        size=1,         # 2D -> keine Sliding-Window-Logik
        group_len=1,
        percentile=99.9,
        dtype=np.float32,
    )
    X_test, Y_test = results["test"]
    clip_val_train = float(meta["clip_val"])
    print(f"[INFO] clip_val_train={clip_val_train:.4g}, shape={X_test.shape}")

    print(f">>> Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print(model.summary())

    print(">>> Vorhersage...")
    pred = model.predict(X_test, batch_size=8, verbose=1)

    print(">>> Frames bauen...")
    frames_all = build_frames_2d(
        low=X_test, pred=pred, high=Y_test,
        p_low=p_low, p_high=p_high, gamma=gamma,
        layout="h", upscale=upscale, pad_px=12,
    )

    out_dir = Path(out_dir)
    out_file = out_dir / f"{Path(model_path).stem}_movie.mp4"
    save_mp4(frames_all, out_file, fps=fps)

    print(">>> Fertig! Film gespeichert unter:")
    print(out_file)

if __name__ == "__main__":
    model_path = Path(__file__).parent / "JENS_IRUNET.hdf5"
    main(
        model_path=model_path,
        data_dir=Path.home() / "data" / "original_data",
        out_dir=Path.home() / "data" / "movies",
        fps=12,
        p_low=1.0, p_high=99.5, gamma=0.8,
        upscale=2,
    )

