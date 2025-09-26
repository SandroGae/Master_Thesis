# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: dl
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import re
import json
from typing import Optional
import numpy as np
import tensorflow as tf
from pathlib import Path
from unet_3d_data import prepare_in_memory_5to5

DATA_ROOT = Path.home() / "data"
EVAL_ROOT = DATA_ROOT  # Output directory

# ===== Anscombe utils (for original-domain metrics) =====
def _sanitize_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (s or "").strip()) or "EVAL"

def _auto_script_name() -> str:
    # Name dieses Eval-Skripts ohne .py
    try:
        path = sys.modules.get("__main__").__file__
    except Exception:
        path = None
    if not path:
        path = sys.argv[0] if sys.argv else "eval"
    return _sanitize_name(os.path.splitext(os.path.basename(path))[0])

def inv_anscombe_tf(z, eps=1e-6):
    """Approximate unbiased inverse of Anscombe; clamp small z for stability."""
    z = tf.maximum(z, eps)
    z2 = z / 2.0
    y = z2**2 - 1.0/8.0 + 1.0/(4.0*z**2) - 11.0/(8.0*z**4)
    return tf.nn.relu(y)

def stable_psnr(y_true, y_pred, eps=1e-12):
    """
    PSNR computed from per-sample MSE with epsilon for stability.
    y_*: Tensor [B, D, H, W, C] (or similar), values expected in [0,1].
    """
    err2 = tf.square(y_true - y_pred)
    axes = tf.range(1, tf.rank(err2))                     # mean over all but batch
    mse = tf.reduce_mean(err2, axis=axes)
    psnr = -10.0 * tf.math.log(mse + eps) / tf.math.log(tf.constant(10.0, dtype=mse.dtype))
    return tf.reduce_mean(psnr)

def compute_ms_ssim(Y_true, Y_pred):
    # (B, D, H, W, C) -> zu (B*D, H, W, C)
    yt2 = tf.reshape(Y_true, (-1, tf.shape(Y_true)[2], tf.shape(Y_true)[3], tf.shape(Y_true)[4]))
    yp2 = tf.reshape(Y_pred, (-1, tf.shape(Y_pred)[2], tf.shape(Y_pred)[3], tf.shape(Y_pred)[4]))
    return float(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0).numpy().mean())

# ===== Selection helpers =====
def pick_checkpoint_dir():
    # sucht unter ~/data nach Verzeichnissen "checkpoints_*"
    if not DATA_ROOT.exists():
        print(f"{DATA_ROOT} existiert nicht.")
        sys.exit(1)
    cand = sorted([p for p in DATA_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith("checkpoints_")])
    if not cand:
        print("Keine Checkpoint-Ordner gefunden unter ~/data (checkpoints_*)")
        sys.exit(1)
    print("Waehle Checkpoint-Ordner:")
    for i, p in enumerate(cand, 1):
        print(f"  [{i}] {p.name}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(cand):
                return cand[idx - 1]

def pick_version(ckpt_dir: Path):
    models = []
    pat = re.compile(r"^V(\d+)_.*\.keras$")
    for p in ckpt_dir.iterdir():
        if p.is_file() and p.suffix == ".keras" and pat.match(p.name):
            models.append(p)
    if not models:
        print(f"Keine V*-Modelle in {ckpt_dir} gefunden.")
        sys.exit(1)
    # sort by V number
    models.sort(key=lambda p: int(p.stem.split('_')[0][1:]))
    print(f"Waehle Modell in {ckpt_dir.name}:")
    for i, p in enumerate(models, 1):
        print(f"  [{i}] {p.name}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(models):
                return models[idx - 1]

# ===== Determine use_vst (from JSON; fallback: folder name) =====
def detect_use_vst(model_path: Path) -> bool:
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return bool(meta.get("data_prep", {}).get("use_vst", False))
        except Exception:
            pass
    return ("anscombe" in model_path.parent.name.lower())


# ===== Test dataset =====
# ÄNDERUNG: Funktion so anpassen, dass sie das Metadaten-Dict zurückgibt
def build_test_dataset(use_vst: bool, size=5, group_len=41, dtype=np.float32, batch_size=4):
    results, meta = prepare_in_memory_5to5( # <-- meta empfangen
        data_dir=Path.home() / "data" / "original_data",
        size=size,
        group_len=group_len,
        use_vst=use_vst,
        dtype=dtype,
    )
    X_test, Y_test = results["test"]
    AUTO = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    ds = ds.batch(batch_size).prefetch(AUTO)
    # ÄNDERUNG: meta zurückgeben
    return ds, X_test.shape[1:], meta

# ===== Collect predictions =====
def collect_preds_and_targets(model, dataset, max_batches=None):
    y_true, y_pred = [], []
    for b, (xb, yb) in enumerate(dataset):
        yhat = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(yhat)
        if max_batches and (b + 1) >= max_batches:
            break
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred

# ===== val_loss helpers fuer Dateinamen =====
def _read_val_loss_from_meta(model_path: Path) -> Optional[float]:
    """Versucht val_loss aus einer nebenliegenden JSON-Meta zu lesen."""
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            if "val_loss" in meta:
                return float(meta["val_loss"])
            if "best_val_loss" in meta:
                return float(meta["best_val_loss"])
            hist = meta.get("history", {})
            if isinstance(hist, dict) and "val_loss" in hist and hist["val_loss"]:
                try:
                    return float(np.min(hist["val_loss"]))
                except Exception:
                    try:
                        return float(hist["val_loss"][-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return None

def _read_val_loss_from_name(model_path: Path) -> Optional[float]:
    """Extrahiert val_loss aus dem Dateinamen (unterstuetzt 'val_loss=' oder 'valloss=')."""
    stem = model_path.stem.lower()
    m = re.search(r"(?:val[_-]?loss|valloss)\s*=\s*([0-9]*\.?[0-9]+)", stem)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _build_eval_filename(model_path: Path, psnr_value: float, val_loss_value: Optional[float], prefix: Optional[str] = None) -> str:
    """
    Baut: <prefix>_<modellstem>_val<loss>_psnr<db>.json
    prefix: Standard = Name des Eval-Skripts.
    """
    stem = model_path.stem
    pref = _sanitize_name(prefix) if prefix else _auto_script_name()
    if val_loss_value is not None:
        return f"{pref}_{stem}_val{val_loss_value:.6f}_psnr{psnr_value:.2f}.json"
    else:
        return f"{pref}_{stem}_psnr{psnr_value:.2f}.json"

# ====== Ergebnisse speichern ======
def save_results(model_path, results: dict):
    # Zielordner: ~/data/model_evaluations
    out_dir = EVAL_ROOT / "model_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # val_loss beschaffen (Meta -> Name -> None)
    val_loss = _read_val_loss_from_meta(model_path)
    if val_loss is None:
        val_loss = _read_val_loss_from_name(model_path)

    # PSNR fuer den Dateinamen
    psnr_value = float(results.get("psnr", 0.0))

    # Dateiname bauen
    out_name = _build_eval_filename(model_path, psnr_value, val_loss)  # prefix auto aus Skriptname
    out_path = out_dir / out_name

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n>> Ergebnisse gespeichert unter: {out_path}")

# ===== Main =====
def main():
    ckpt_dir = pick_checkpoint_dir()
    model_path = pick_version(ckpt_dir)

    use_vst = detect_use_vst(model_path)
    print(f"\n>> Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(">> Baue Test-Dataset… (use_vst =", use_vst, ")")
    # ÄNDERUNG: meta hier empfangen
    test_ds, input_shape, data_meta = build_test_dataset(
        use_vst=use_vst, size=5, group_len=41, dtype=np.float32, batch_size=4
    )

    Y_true, Y_pred = collect_preds_and_targets(model, test_ds, max_batches=None)
    
    if use_vst:
        clip_val = data_meta.get("clip_val")
        if clip_val is None:
            print("[FEHLER] clip_val nicht in data_meta gefunden. Abbruch.")
            sys.exit(1)
            
        print(f">> Wende inverse Transformation an mit clip_val = {clip_val:.4f}")
        
        # 1. Skalierung rückgängig machen, um rohe VST-Werte zu erhalten
        Y_true_unscaled = tf.convert_to_tensor(Y_true) * clip_val
        Y_pred_unscaled = tf.convert_to_tensor(Y_pred) * clip_val
        
        # 2. Inverse Transformation auf unskalierten Werten anwenden
        # HINWEIS: Die Normalisierung des Originalraums (z.B. durch einen anderen clip_val)
        # ist hier implizit, da die inv_anscombe-Funktion Werte im "Zähl"-Raum erzeugt,
        # die aber durch die ursprüngliche Normalisierung in `unet_3d_data` bereits
        # so skaliert sind, dass sie nach der Rücktransformation wieder grob im Bereich [0,1] liegen sollten.
        # Für eine 100% saubere Trennung müsste man den clip_val für den VST-Raum und
        # den clip_val für den Original-Raum getrennt berechnen und übergeben.
        # Für den Moment ist diese Vereinfachung aber der richtige Weg.
        
        Y_true_o = inv_anscombe_tf(Y_true_unscaled)
        Y_pred_o = inv_anscombe_tf(Y_pred_unscaled)
        
        # 3. Clipping zur Stabilisierung
        Y_true_o = tf.clip_by_value(Y_true_o, 0.0, 1.0)
        Y_pred_o = tf.clip_by_value(Y_pred_o, 0.0, 1.0)
        
        Y_true_m = Y_true_o.numpy()
        Y_pred_m = Y_pred_o.numpy()
    else:
        Y_true_m = Y_true
        Y_pred_m = Y_pred

    yt = Y_true_m.ravel()
    yp = Y_pred_m.ravel()
    mse  = float(np.mean((yt - yp) ** 2))
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))

    psnr = float(tf.image.psnr(Y_true_m, Y_pred_m, max_val=1.0).numpy().mean())
    # Mittel-Slice vorbereitet (optional nutzbar)
    Y_true_2d = Y_true_m[:, Y_true_m.shape[1] // 2, :, :, :]
    Y_pred_2d = Y_pred_m[:, Y_pred_m.shape[1] // 2, :, :, :]
    ms_ssim = compute_ms_ssim(Y_true_m, Y_pred_m)

    print("\n=== Evaluation auf Test Set (Originalraum) ===")
    print(f"Modell       : {model_path.name}")
    print(f"INPUT_SHAPE  : {input_shape}")
    print(f"use_vst      : {use_vst}")
    print(f"MSE          : {mse:.6f}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"PSNR         : {psnr:.2f} dB")
    print(f"MS-SSIM      : {ms_ssim:.4f}")

    # ---- Ergebnisse abspeichern ----
    results = {
        "model": model_path.name,
        "input_shape": tuple(int(x) for x in input_shape),
        "use_vst": use_vst,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ms_ssim": ms_ssim,
    }
    save_results(model_path, results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")

