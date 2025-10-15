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
# MOVIE_unet_3d.py
# Baut ein MP4: Low-count | Denoised (Model) | High-count, mittleres Slice je 5er-Fenster.
# Nutzt exakt dieselbe ALT-Pipeline (kein VST) wie im Training.

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from unet_3d_data import prepare_in_memory

# ---------- Anzeige-Helfer ----------
def stretch_with_window(x01, lo, hi, gamma=0.8):
    x = np.clip(x01.astype(np.float32), 0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max() if x.max() > x.min() else 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    if gamma and gamma > 0: x = np.power(x, gamma)
    return (x*255.0 + 0.5).astype(np.uint8)

def make_rgb_labeled(panel_gray, label_text, pad=6):
    h, w = panel_gray.shape
    bar_h = 26
    canvas = Image.new("RGB", (w, h + bar_h + pad), (0, 0, 0))
    panel_img = Image.fromarray(panel_gray).convert("RGB")
    canvas.paste(panel_img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
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
def build_frames(low, pred, high, *, p_low=1.0, p_high=99.5, gamma=0.8, upscale=2, pad_px=12):
    frames=[]
    for i in range(low.shape[0]):
        # gemeinsames Fenster – nimm High (oder die Vereinigung)
        # Option A: nur High als Referenz
        lo, hi = np.percentile(high[i,...,0], [p_low, p_high])
        # Option B (streng gleich): Vereinigung aller drei
        # all_vals = np.concatenate([low[i,...,0].ravel(), pred[i,...,0].ravel(), high[i,...,0].ravel()])
        # lo, hi = np.percentile(all_vals, [p_low, p_high])

        l8 = stretch_with_window(low [i,...,0], lo, hi, gamma)
        p8 = stretch_with_window(pred[i,...,0], lo, hi, gamma)
        h8 = stretch_with_window(high[i,...,0], lo, hi, gamma)
        l = make_rgb_labeled(l8,"Low-count")
        p = make_rgb_labeled(p8,"Denoised (Model)")
        h = make_rgb_labeled(h8,"High-count")
        frames.append(upscale_rgb(hstack_same_height([l,p,h], pad_px=pad_px), scale=upscale))
    return frames

def build_center_frames(low, pred, high, *, size=5, group_len=41, group_index=0,
                        p_low=1.0, p_high=99.5, gamma=0.8, layout="h",
                        upscale=2, pad_px=12):
    # low/pred/high: (B, D, H, W, 1)
    assert low.shape == pred.shape == high.shape
    B, D, H, W, C = low.shape
    assert C == 1 and size % 2 == 1
    k = size // 2
    windows_per_group = group_len - size + 1  # 37
    num_groups = B // windows_per_group
    if not (0 <= group_index < num_groups):
        raise ValueError(f"group_index {group_index} out of range [0,{num_groups-1}]")
    start = group_index * windows_per_group
    end   = start + windows_per_group

    frames = []
    for b in range(start, end):
        l = low [b, k, ..., 0]   # (H,W)
        p = pred[b, k, ..., 0]
        h = high[b, k, ..., 0]

        # gemeinsames Fenster pro Triple (vom High-Frame)
        lo, hi = np.percentile(h, [p_low, p_high])
        l8 = stretch_with_window(l, lo, hi, gamma)
        p8 = stretch_with_window(p, lo, hi, gamma)
        h8 = stretch_with_window(h, lo, hi, gamma)

        l_rgb = make_rgb_labeled(l8, "Low-count")
        p_rgb = make_rgb_labeled(p8, "Denoised (Model)")
        h_rgb = make_rgb_labeled(h8, "High-count")

        frame_rgb = hstack_same_height([l_rgb, p_rgb, h_rgb], pad_px=pad_px) if layout=="h" \
                    else np.concatenate([l_rgb, p_rgb, h_rgb], axis=0)
        frames.append(upscale_rgb(frame_rgb, scale=upscale))
    return frames

# ---------- Speichern ----------
def save_mp4(frames_rgb, out_path, fps=12):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=fps, quality=8) as w:
        for fr in frames_rgb:
            w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

# ---------- Denormalisieren (optional) ----------
def denorm_counts(x01, clip_val, dtype=np.uint16):
    x = np.clip(x01, 0.0, 1.0) * float(clip_val)
    if dtype is not None:
        x = np.rint(x).astype(dtype)
    return x

# ---------- Main ----------
def main(
    model_path: Path,
    data_dir: Path = Path.home() / "data" / "original_data",
    size: int = 5,
    group_len: int = 41,
    out_dir: Path = Path.home() / "data" / "movies",
    group_index: int = 0,
    fps: int = 12,
    save_counts: bool = False,
    spacer_seconds=0.25, spacer_color=(0,0,0)
):
    print(">>> Lade Testdaten (ALT-Pipeline, kein VST)...")
    (results, meta) = prepare_in_memory(
        data_dir=data_dir,
        size=size,
        group_len=group_len,
        percentile=99.9,
        dtype=np.float32,
    )
    X_test, Y_test = results["test"]
    clip_val_train = float(meta["clip_val"])
    print(f"[INFO] clip_val_train={clip_val_train:.4g}, size={size}, group_len={group_len}")

    print(f">>> Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(">>> Vorhersage...")
    pred = model.predict(X_test, batch_size=8, verbose=1)

    print(">>> Frames bauen (20 Serien mit Trenner)...")
    # Wie viele Fenster/Gruppe und wie viele Gruppen gibt's im Testset?
    windows_per_group = group_len - size + 1  # 37 bei 41/5
    B = X_test.shape[0]
    num_groups_total = B // windows_per_group
    use_groups = min(20, num_groups_total)  # die ersten 20

    # Erstmal eine Serie bauen, um die Frame-Größe zu kennen
    frames_first = build_center_frames(
        low=X_test, pred=pred, high=Y_test,
        size=size, group_len=group_len, group_index=0,
        # Sichtbarkeit/Look:
        p_low=1.0, p_high=99.5, gamma=0.8,
        layout="h", upscale=2, pad_px=12,
    )

    # Falls leer (sollte nie passieren), abbrechen
    if not frames_first:
        raise RuntimeError("Keine Frames erzeugt – prüfe Eingaben/Shapes.")

    # Grünen Spacer vorbereiten (0.5s bei gegebener FPS)
    fps = int(fps) if fps else 10
    spacer_frames = max(1, int(round(spacer_seconds * fps)))
    H, W, _ = frames_first[0].shape
    spacer = np.zeros((H, W, 3), dtype=np.uint8)
    spacer[..., 0] = spacer_color[0]
    spacer[..., 1] = spacer_color[1]
    spacer[..., 2] = spacer_color[2]

    # Alle Frames sammeln
    frames_all = []
    for g in range(use_groups):
        if g == 0:
            frames_all.extend(frames_first)
        else:
            frames_g = build_center_frames(
                low=X_test, pred=pred, high=Y_test,
                size=size, group_len=group_len, group_index=g,
                p_low=1.0, p_high=99.5, gamma=0.8,
                layout="h", upscale=2, pad_px=12,
            )
            frames_all.extend(frames_g)

        # Nach jeder Serie (außer der letzten) Spacer einfügen
        if g < use_groups - 1:
            for _ in range(spacer_frames):
                frames_all.append(spacer)

    # Speichern
    out_dir = Path(out_dir)
    stem = Path(model_path).with_suffix("").name
    save_mp4(frames_all, out_dir / f"{stem}_groups0to{use_groups-1}_center.mp4", fps=fps)

    if save_counts:
        print(">>> Speichere denoiste Original-Counts (npy)...")
        pred_counts = denorm_counts(pred, clip_val_train, dtype=np.uint16)
        np.save(out_dir / f"{stem}_pred_counts.npy", pred_counts)
        print(f"[OK] counts npy: {out_dir / (stem + '_pred_counts.npy')}")

if __name__ == "__main__":
    model_path = Path.home() / "data" / "checkpoints_3d_unet" / "unet_3d_V1_valloss_1.065e-02_PSNR_42.keras"
    main(
        model_path=model_path,
        data_dir=Path.home() / "data" / "original_data",
        size=5,
        group_len=41,
        out_dir=Path.home() / "data" / "movies",
        group_index=0,
        fps=12,
        save_counts=True,
    )

