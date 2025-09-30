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
# evaluation.py
import os, sys, re, json
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import tensorflow as tf

from unet_3d_data import prepare_in_memory_5to5  # <- deine Datenfunktion

DATA_ROOT = Path.home() / "data"
EVAL_ROOT = DATA_ROOT  # Ausgabeverzeichnis für Ergebnisse

# ========== Helfer ==========
def _sanitize_name(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s) or "EVAL"

def _auto_script_name() -> str:
    try:
        path = sys.modules.get("__main__").__file__
    except Exception:
        path = None
    if not path:
        path = sys.argv[0] if sys.argv else "eval"
    return _sanitize_name(os.path.splitext(os.path.basename(path))[0])

def compute_ms_ssim(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    # (B, D, H, W, C) -> (B*D, H, W, C)
    yt2 = tf.reshape(tf.convert_to_tensor(Y_true), (-1, Y_true.shape[2], Y_true.shape[3], Y_true.shape[4]))
    yp2 = tf.reshape(tf.convert_to_tensor(Y_pred), (-1, Y_pred.shape[2], Y_pred.shape[3], Y_pred.shape[4]))
    ms = tf.image.ssim_multiscale(yt2, yp2, max_val=1.0)
    return float(tf.reduce_mean(ms).numpy())

# ========== Auswahl der Dateien ==========
def pick_checkpoint_dir() -> Path:
    if not DATA_ROOT.exists():
        print(f"{DATA_ROOT} existiert nicht."); sys.exit(1)
    cand = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("checkpoints_")])
    if not cand:
        print("Keine Checkpoint-Ordner unter ~/data gefunden (checkpoints_*)"); sys.exit(1)

    print("Wähle Checkpoint-Ordner:")
    for i, p in enumerate(cand, 1):
        print(f"  [{i}] {p.name}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(cand):
                return cand[idx - 1]

def pick_version(ckpt_dir: Path) -> Path:
    """
    Listet alle .keras-Modelle, die irgendwo im Namen ein 'V<Zahl>' enthalten.
    Beispiel: V1_..., unet3d_V12_valloss..., fooV3bar.keras
    """
    pat = re.compile(r"V(\d+)", re.IGNORECASE)
    models = [p for p in ckpt_dir.iterdir()
              if p.is_file() and p.suffix == ".keras" and pat.search(p.name)]
    if not models:
        print(f"Keine Modelle mit 'V<Zahl>' im Namen in {ckpt_dir} gefunden.")
        sys.exit(1)

    # Sortieren nach der ersten gefundenen V-Zahl
    def extract_vnum(name: str) -> int:
        m = pat.search(name)
        return int(m.group(1)) if m else 999999
    models.sort(key=lambda p: extract_vnum(p.name))

    print(f"Wähle Modell in {ckpt_dir.name}:")
    for i, p in enumerate(models, 1):
        print(f"  [{i}] {p.name}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(models):
                return models[idx - 1]


# ========== Test-Dataset ==========
def build_test_dataset(size=5, group_len=41, dtype=np.float32, batch_size=4) -> Tuple[tf.data.Dataset, Tuple[int,int,int,int]]:
    results = prepare_in_memory_5to5(
        data_dir=Path.home() / "data" / "original_data",
        size=size,
        group_len=group_len,
        dtype=dtype,
    )
    X_test, Y_test = results["test"]
    ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, X_test.shape[1:]  # (D,H,W,C)

# ========== Vorhersagen sammeln ==========
def collect_preds_and_targets(model, dataset, max_batches=None):
    y_true, y_pred = [], []
    for b, (xb, yb) in enumerate(dataset):
        yhat = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(yhat)
        if max_batches and (b + 1) >= max_batches:
            break
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)

# ========== val_loss aus Meta/Name lesen (optional für Dateinamen) ==========
def _read_val_loss_from_meta(model_path: Path) -> Optional[float]:
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            if "val_loss" in meta: return float(meta["val_loss"])
            if "best_val_loss" in meta: return float(meta["best_val_loss"])
            hist = meta.get("history", {})
            if isinstance(hist, dict) and "val_loss" in hist and hist["val_loss"]:
                try:
                    return float(np.min(hist["val_loss"]))
                except Exception:
                    return float(hist["val_loss"][-1])
    except Exception:
        pass
    return None

def _read_val_loss_from_name(model_path: Path) -> Optional[float]:
    stem = model_path.stem.lower()
    m = re.search(r"(?:val[_-]?loss|valloss)\s*=?\s*([0-9]*\.?[0-9]+)", stem)
    if m:
        try: return float(m.group(1))
        except Exception: return None
    return None

def _build_eval_filename(model_path: Path, psnr_value: float, val_loss_value: Optional[float], prefix: Optional[str] = None) -> str:
    stem = model_path.stem
    pref = _sanitize_name(prefix) if prefix else _auto_script_name()
    if val_loss_value is not None:
        return f"{pref}_{stem}_val{val_loss_value:.6f}_psnr{psnr_value:.2f}.json"
    else:
        return f"{pref}_{stem}_psnr{psnr_value:.2f}.json"

# ========== Ergebnisse speichern ==========
def save_results(model_path: Path, results: dict):
    out_dir = EVAL_ROOT / "model_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_loss = _read_val_loss_from_meta(model_path)
    if val_loss is None:
        val_loss = _read_val_loss_from_name(model_path)
    psnr_value = float(results.get("psnr", 0.0))
    out_name = _build_eval_filename(model_path, psnr_value, val_loss)
    out_path = out_dir / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>> Ergebnisse gespeichert unter: {out_path}")

# ========== Main ==========
def main():
    ckpt_dir = pick_checkpoint_dir()
    model_path = pick_version(ckpt_dir)

    print(f"\n>> Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(">> Baue Test-Dataset…")
    test_ds, input_shape = build_test_dataset(size=5, group_len=41, dtype=np.float32, batch_size=4)

    Y_true, Y_pred = collect_preds_and_targets(model, test_ds, max_batches=None)

    # Direkt im [0,1]-Raum auswerten (keine VST/Anscombe mehr)
    Y_true_m = Y_true
    Y_pred_m = Y_pred

    yt = Y_true_m.ravel()
    yp = Y_pred_m.ravel()
    mse  = float(np.mean((yt - yp) ** 2))
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))
    psnr = float(tf.image.psnr(Y_true_m, Y_pred_m, max_val=1.0).numpy().mean())
    ms_ssim = compute_ms_ssim(Y_true_m, Y_pred_m)

    print("\n=== Evaluation auf Test Set ===")
    print(f"Modell       : {model_path.name}")
    print(f"INPUT_SHAPE  : {input_shape}")
    print(f"MSE          : {mse:.6f}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"PSNR         : {psnr:.2f} dB")
    print(f"MS-SSIM      : {ms_ssim:.4f}")

    results = {
        "model": model_path.name,
        "input_shape": tuple(int(x) for x in input_shape),
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

