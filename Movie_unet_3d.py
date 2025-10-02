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
# eval_make_movie_unet3d.py
# Evaluierung/Film fuer ALT-Pipeline (kein VST). Nutzt exakt die gleiche Datenaufbereitung
# wie dein Training (prepare_in_memory). Zeigt Film low | pred | high (mittleres Slice).
# Optional: speichere denoiste Original-Counts (vor Normalisierung) ab.

import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf

from unet_3d_data import prepare_in_memory

# ---------- Anzeige-Helfer ----------
def to_uint8_global(img01, vmin=0.0, vmax=1.0):
    """[0,1] -> uint8 mit fester globaler Skalierung (physikalisch sinnvoller Vergleich ueber Zeit)."""
    x = np.clip((img01 - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def to_uint8_local(img01, p_low=1.0, p_high=99.0):
    """[0,1] -> uint8 mit per-Frame-Autokontrast (visuell knackig, aber Helligkeit nicht vergleichbar)."""
    x = np.clip(img01, 0.0, 1.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = x.min(), x.max() if x.max() > x.min() else (0.0, 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

# ---------- Frames bauen ----------
def build_center_frames(low, pred, high, size=5, group_len=41, group_index=0,
                        contrast="global"):
    """
    low/pred/high: Arrays (B, D, H, W, 1) in [0,1]
    Liefert Liste von (H*3, W)-uint8-Frames fuer genau EINE 41er-Gruppe (-> 37 Fenster).
    contrast: "global" oder "local"
    """
    assert low.shape == pred.shape == high.shape, "low/pred/high shape mismatch"
    B, D, H, W, C = low.shape
    assert C == 1, "erwarte Kanal=1"

    k = size // 2
    windows_per_group = group_len - size + 1  # 37 bei 41/5
    num_groups = B // windows_per_group
    if group_index < 0 or group_index >= num_groups:
        raise ValueError(f"group_index {group_index} out of range [0,{num_groups-1}]")

    start = group_index * windows_per_group
    end   = start + windows_per_group

    # Waehle Mapping-Funktion
    if contrast == "global":
        map_fn = to_uint8_global
    elif contrast == "local":
        map_fn = to_uint8_local
    else:
        raise ValueError("contrast must be 'global' or 'local'")

    frames = []
    for b in range(start, end):
        l = low [b, k, ..., 0]
        p = pred[b, k, ..., 0]
        h = high[b, k, ..., 0]
        l8 = map_fn(l); p8 = map_fn(p); h8 = map_fn(h)
        frame = np.concatenate([l8, p8, h8], axis=0)  # (H*3, W)
        frames.append(frame)
    return frames

# ---------- Speichern ----------
def save_mp4(frames_gray, out_path, fps=10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path.as_posix(), fps=fps) as w:
        for fr in frames_gray:
            w.append_data(fr)
    print(f"[OK] MP4: {out_path}")

def save_gif(frames_gray, out_path, fps=10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path.as_posix(), frames_gray, duration=1.0/max(fps,1), loop=0)
    print(f"[OK] GIF: {out_path}")

# ---------- Denormalisieren zu originalen Counts ----------
def denorm_counts(x01, clip_val, dtype=np.uint16):
    """
    x01 in [0,1] -> Originalraum (ungefaehr 'counts').
    Achtung: trainingsbedingtes Clipping bei clip_val bleibt inhärent.
    """
    x = np.clip(x01, 0.0, 1.0) * float(clip_val)
    if dtype is not None:
        x = np.rint(x).astype(dtype)
    return x

# ---------- Main-Flow ----------
def main(
    model_path: Path,
    data_dir: Path = Path.home() / "data" / "original_data",
    size: int = 5,
    group_len: int = 41,
    out_dir: Path = Path.home() / "data" / "movies",
    group_index: int = 0,
    fps: int = 10,
    contrast: str = "global",   # "global" empfohlen fuer fairen Vergleich
    save_counts: bool = False,  # True => speichere denoiste Original-Counts (npy)
):
    print(">>> Lade Testdaten mit ALT-Pipeline (kein VST)...")
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

    print(">>> Frames bauen...")
    frames = build_center_frames(
        low=X_test, pred=pred, high=Y_test,
        size=size, group_len=group_len, group_index=group_index,
        contrast=contrast,
    )

    out_dir = Path(out_dir)
    stem = Path(model_path).with_suffix("").name
    save_mp4(frames, out_dir / f"{stem}_group{group_index}_center.mp4", fps=fps)
    # optional GIF:
    # save_gif(frames, out_dir / f"{stem}_group{group_index}_center.gif", fps=fps)

    if save_counts:
        print(">>> Speichere denoiste Original-Counts (npy)...")
        pred_counts = denorm_counts(pred, clip_val_train, dtype=np.uint16)
        np.save(out_dir / f"{stem}_pred_counts.npy", pred_counts)
        print(f"[OK] counts npy: {out_dir / (stem + '_pred_counts.npy')}")

if __name__ == "__main__":
    # ---- Beispielaufruf anpassen ----
    model_path = Path.home() / "data" / "checkpoints_3d_unet" / "unet_3d_V1_valloss_1.065e-02_PSNR_42.keras"
    main(
        model_path=model_path,
        data_dir=Path.home() / "data" / "original_data",
        size=5,
        group_len=41,
        out_dir=Path.home() / "data" / "movies",
        group_index=0,
        fps=10,
        contrast="global",      # fuer faire Vergleiche
        save_counts=True,       # auf Wunsch Original-Counts ablegen
    )

