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
# Movie_Universal.py
# Einheitlicher Movie-Generator fuer 5-Stack-Modelle (D,H,W,1),
# mit Pipeline-Adaptern pro Code-Namen (z. B. "unet_3d", "unet_3d_JENS_mae").
# Visualisierung ist immer identisch (Low | Denoised | High), unabhaengig von der Normalisierung.

from pathlib import Path
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# =========================
# Anzeige-/Video-Helfer
# =========================

def _clip01(x):
    return np.clip(x.astype(np.float32), 0.0, 1.0)

def stretch_with_window(x01, lo, hi, gamma=0.8):
    """x01 in [0,1]; wendet gemeinsames Fenster [lo,hi] (aus High) an und optional Gamma."""
    x = _clip01(x01)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max() if x.max() > x.min() else 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    if gamma and gamma > 0:
        x = np.power(x, gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)

def make_rgb_labeled(panel_gray, label_text, pad=6):
    """Schwarze Titelzeile + Graubild, Rueckgabe als RGB-Array."""
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

def save_mp4(frames_rgb, out_path, fps=12):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=int(fps), quality=8) as w:
        for fr in frames_rgb:
            w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

# =========================
# Frame-Bau (einheitlich)
# =========================

def build_center_frames(
    low, pred, high, *,
    size=5, group_len=41, group_index=0,
    p_low=1.0, p_high=99.5, gamma=0.8,
    layout="h", upscale=2, pad_px=12,
    labels=("Low-count", "Denoised (Model)", "High-count"),
    energy_match=True,
):
    """
    low, pred, high: (B, D, H, W, 1), Werte erwartet in [0,1] (werden geclippt).
    Visualisierung: pro Fenster zentriertes Slice (k=size//2), Fensterung aus HIGH,
    optionales Energiematching pred->high sum.
    """
    assert low.shape == pred.shape == high.shape
    B, D, H, W, C = low.shape
    assert C == 1, "Erwarte Kanal = 1"
    k = size // 2
    windows_per_group = group_len - size + 1
    num_groups = B // windows_per_group
    if group_index < 0 or group_index >= num_groups:
        raise ValueError(f"group_index {group_index} out of range [0,{num_groups-1}]")
    start = group_index * windows_per_group
    end   = start + windows_per_group

    frames = []
    for b in range(start, end):
        l = _clip01(low [b, k, ..., 0])
        p = _clip01(pred[b, k, ..., 0])
        h = _clip01(high[b, k, ..., 0])

        # Energiematching: gleiche Gesamt-Helligkeit wie High (robust gegen Pipeline-Scale)
        if energy_match:
            num = float(h.sum())
            den = float(p.sum())
            scale = num / max(den, 1e-8)
            p_eq = _clip01(p * scale)
        else:
            p_eq = p

        # Gemeinsame Fensterung aus HIGH
        lo, hi = np.percentile(h, [p_low, p_high])
        l8 = stretch_with_window(l,    lo, hi, gamma)
        p8 = stretch_with_window(p_eq, lo, hi, gamma)
        h8 = stretch_with_window(h,    lo, hi, gamma)

        l_rgb = make_rgb_labeled(l8, labels[0])
        p_rgb = make_rgb_labeled(p8, labels[1])
        h_rgb = make_rgb_labeled(h8, labels[2])

        if layout == "h":
            frame_rgb = hstack_same_height([l_rgb, p_rgb, h_rgb], pad_px=pad_px)
        else:
            frame_rgb = np.concatenate([l_rgb, p_rgb, h_rgb], axis=0)

        frames.append(upscale_rgb(frame_rgb, scale=upscale))
    return frames

# ======================================
# Pipeline-Adapter (Registry-Mechanik)
# ======================================

def _batch_map_sumscale_numpy(X, Y, normalizer, batch=64):
    """
    Fuehrt Jens' SumScaleNormalizer (map) in Batches auf Numpy-Daten aus
    und gibt Numpy zurueck. Erwartet X,Y in (B,D,H,W,1) float.
    """
    outs_x, outs_y = [], []
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = tf.convert_to_tensor(X[i:i+batch], dtype=tf.float32)
        yb = tf.convert_to_tensor(Y[i:i+batch], dtype=tf.float32)
        xn, yn = normalizer.map(xb, yb)
        outs_x.append(xn.numpy())
        outs_y.append(yn.numpy())
    Xn = np.concatenate(outs_x, axis=0)
    Yn = np.concatenate(outs_y, axis=0)
    return Xn, Yn

# ---- Adapter: ALT-Pipeline aus unet_3d (kein VST, globales Clip auf [0,1]) ----
def unet_3d_loader(data_dir, size, group_len, dtype=np.float32):
    """
    Laedt test-Set bereits korrekt skaliert (ALT_no_VST) aus unet_3d_data.prepare_in_memory.
    Gibt X_test, Y_test, meta zurueck (meta["clip_val"] wird ggf. fuer Counts-Export genutzt).
    """
    from unet_3d_data import prepare_in_memory
    (results, meta) = prepare_in_memory(
        data_dir=Path(data_dir),
        size=size,
        group_len=group_len,
        percentile=99.9,
        dtype=dtype
    )
    X_test, Y_test = results["test"]
    return X_test.astype(np.float32), Y_test.astype(np.float32), meta

# ---- Adapter: Jens-Pipeline (SumScaleNormalizer auf Inferenz anwenden) ----
def unet_3d_JENS_mae_loader(data_dir, size, group_len, dtype=np.float32):
    """
    Laedt Rohdaten und normalisiert sie fuer Inferenz exakt wie im Jens-Valid/Test:
      SumScaleNormalizer(scale_range=[5000,5001], normalize_label=True,
      axis=(1,2,3), batch_mode=True, clip_before=[0,inf], clip_after=[0,1])
    """
    from unet_3d_data_JENS import prepare_in_memory_5to5
    from jens_stuff import SumScaleNormalizer

    results = prepare_in_memory_5to5(
        data_dir=Path(data_dir),
        size=size, group_len=group_len, dtype=dtype
    )
    X_test, Y_test = results["test"]  # Roh-Counts (N,H,W) bereits in (B,D,H,W,1) Form

    norm_valid = SumScaleNormalizer(
        scale_range=[5000, 5001], pre_offset=0.0, normalize_label=True,
        axis=(1,2,3), batch_mode=True, clip_before=[0., float("inf")], clip_after=[0.,1.]
    )
    Xn, Yn = _batch_map_sumscale_numpy(X_test, Y_test, norm_valid, batch=64)
    meta = {"pipeline": "JENS_SumScale", "scale_range": [5000,5001], "dtype": str(dtype)}
    return Xn.astype(np.float32), Yn.astype(np.float32), meta

# Registry der bekannten Pipelines: key = Code-Name, value = Loader-Funktion
PIPELINE_REGISTRY = {
    "unet_3d": unet_3d_loader,
    "unet_3d_JENS_mae": unet_3d_JENS_mae_loader,
    # -> fuege hier zukuenftige Loader ein: "mein_code": my_loader_func
}

# ======================================
# Hauptfunktion
# ======================================

def make_movie_for_code(
    code_name: str,
    model_path: Path,
    data_dir: Path,
    *,
    size: int = 5,
    group_len: int = 41,
    out_dir: Path = None,
    start_group: int = 0,
    max_groups: int = 20,
    fps: int = 12,
    gamma: float = 0.8,
    p_low: float = 1.0,
    p_high: float = 99.5,
    layout: str = "h",
    upscale: int = 2,
    pad_px: int = 12,
    spacer_seconds: float = 0.25,
    spacer_color=(0,0,0),
    energy_match: bool = True,
    save_counts: bool = False,   # nur sinnvoll, wenn Clip/Counts definiert sind
):
    """
    code_name: muss in PIPELINE_REGISTRY enthalten sein (z. B. "unet_3d" oder "unet_3d_JENS_mae").
    model_path: .keras/.h5 Pfad.
    Visualisierung ist pipeline-unabhaengig identisch.
    """
    if code_name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unbekannter code_name '{code_name}'. Bekannte: {list(PIPELINE_REGISTRY.keys())}")

    data_dir = Path(data_dir)
    model_path = Path(model_path)
    if out_dir is None:
        out_dir = Path.home() / "data" / "movies"
    out_dir = Path(out_dir)

    # 1) Daten laden & pipeline-spezifisch normalisieren
    print(f">>> [Loader] Code={code_name} | data_dir={data_dir}")
    X_test, Y_test, meta = PIPELINE_REGISTRY[code_name](data_dir, size, group_len, np.float32)

    # 2) Modell laden
    print(f">>> Lade Modell: {model_path}")
    # compile=False, da nur Inferenz
    model = tf.keras.models.load_model(model_path, compile=False)

    # 3) Vorhersage
    print(">>> Vorhersage...")
    pred = model.predict(X_test, batch_size=8, verbose=1)

    # 4) Frames bauen
    windows_per_group = group_len - size + 1
    B = X_test.shape[0]
    total_groups = B // windows_per_group
    use_groups = max(0, min(max_groups, total_groups - start_group))
    if use_groups == 0:
        raise RuntimeError(f"Keine gueltigen Gruppen im Testset. start_group={start_group}, total_groups={total_groups}")

    print(f">>> Frames bauen: groups {start_group}..{start_group+use_groups-1} (je {windows_per_group} Fenster)")

    frames_all = []
    # erste Gruppe bauen, um Groesse zu fixieren
    frames_g0 = build_center_frames(
        low=X_test, pred=pred, high=Y_test,
        size=size, group_len=group_len, group_index=start_group,
        p_low=p_low, p_high=p_high, gamma=gamma,
        layout=layout, upscale=upscale, pad_px=pad_px,
        energy_match=energy_match
    )
    if not frames_g0:
        raise RuntimeError("Keine Frames erzeugt – pruefe Eingaben/Shapes.")

    # Spacer
    fps = int(fps) if fps else 10
    spacer_frames = max(1, int(round(spacer_seconds * fps)))
    H, W, _ = frames_g0[0].shape
    spacer = np.zeros((H, W, 3), dtype=np.uint8)
    spacer[..., 0] = spacer_color[0]
    spacer[..., 1] = spacer_color[1]
    spacer[..., 2] = spacer_color[2]

    frames_all.extend(frames_g0)

    for gi in range(start_group + 1, start_group + use_groups):
        frames_g = build_center_frames(
            low=X_test, pred=pred, high=Y_test,
            size=size, group_len=group_len, group_index=gi,
            p_low=p_low, p_high=p_high, gamma=gamma,
            layout=layout, upscale=upscale, pad_px=pad_px,
            energy_match=energy_match
        )
        frames_all.extend(frames_g)
        # Trenner (nicht nach der letzten Gruppe)
        if gi < (start_group + use_groups - 1):
            frames_all.extend([spacer] * spacer_frames)

    # 5) MP4 speichern
    stem = f"{model_path.with_suffix('').name}__{code_name}_groups{start_group}to{start_group+use_groups-1}_center"
    out_mp4 = out_dir / f"{stem}.mp4"
    save_mp4(frames_all, out_mp4, fps=fps)

    # 6) Optional: Denoised-Counts speichern (nur sinnvoll, wenn Clip/Counts bekannt)
    if save_counts:
        clip_val = None
        # bei ALT: meta["clip_val"] vorhanden; bei Jens: typischerweise nicht sinnvoll fuer echte Counts
        if isinstance(meta, dict):
            clip_val = meta.get("clip_val", None)
        if clip_val is not None and np.isfinite(clip_val) and clip_val > 0:
            print(">>> Speichere denoiste Counts (ALT-Clip-Domain)...")
            pred_counts = np.clip(pred, 0.0, 1.0) * float(clip_val)
            pred_counts = np.rint(pred_counts).astype(np.uint16)
            np.save(out_dir / f"{stem}_pred_counts.npy", pred_counts)
            print(f"[OK] counts npy: {out_dir / (stem + '_pred_counts.npy')}")
        else:
            print("[HINWEIS] save_counts=True, aber kein gueltiger clip_val verfuegbar – ueberspringe Counts-Export.")

    print("[OK] Fertig.")

# =========================
# Beispiel-Aufrufe
# =========================
if __name__ == "__main__":
    make_movie_for_code(
        code_name="unet_3d",
        model_path=Path.home() / "data" / "checkpoints_3d_unet" /
                    "unet_3d_V1_valloss_1.065e-02_PSNR_42.keras",
        data_dir=Path.home() / "data" / "original_data",
        size=5, group_len=41,
        out_dir=Path.home() / "data" / "movies",
        start_group=0, max_groups=20,
        fps=12, gamma=0.8, p_low=1.0, p_high=99.5,
        layout="h", upscale=2, pad_px=12,
        energy_match=True,
        save_counts=True
    )

    make_movie_for_code(
        code_name="unet_3d_JENS_mae",
        model_path=Path.home() / "data" / "checkpoints_3d_unet" /
                    "unet_3d_JENS_mae_V1_valloss_9.9e-03_PSNR_43.keras",
        data_dir=Path.home() / "data" / "original_data",
        size=5, group_len=41,
        out_dir=Path.home() / "data" / "movies",
        start_group=0, max_groups=20,
        fps=12, gamma=0.8, p_low=1.0, p_high=99.5,
        layout="h", upscale=2, pad_px=12,
        energy_match=True,
        save_counts=False
    )


