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
# Evaluation.py
import os
import sys
import re
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# --- R^2 with sklearn (fallback if not installed) ---
try:
    from sklearn.metrics import r2_score as _sk_r2
    def r2_score(y_true, y_pred):
        return _sk_r2(y_true, y_pred)
except Exception:
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

# --- your data pipeline ---
from unet_3d_data import prepare_in_memory_5to5

DATA_ROOT = Path.home() / "data"

# ===== Anscombe utils (for original-domain metrics) =====
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

# ===== Selection helpers =====
def pick_checkpoint_dir():
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
        if s.isdigit() and 1 <= int(s) <= len(cand):
            return cand[int(s) - 1]

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
        if s.isdigit() and 1 <= int(s) <= len(models):
            return models[int(s) - 1]

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
def build_test_dataset(use_vst: bool, size=5, group_len=41, dtype=np.float32, batch_size=4):
    results, _ = prepare_in_memory_5to5(
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
    return ds, X_test.shape[1:]

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

# ====== Ergebnisse speichern ======
def save_results(model_path, results: dict):
    out_dir = Path("model_evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)  # Ordner anlegen falls nicht da

    # gleicher Grundname wie Modell, aber .json statt .keras
    out_path = out_dir / (model_path.stem + "_evaluation.json")

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
    test_ds, input_shape = build_test_dataset(
        use_vst=use_vst, size=5, group_len=41, dtype=np.float32, batch_size=4
    )

    Y_true, Y_pred = collect_preds_and_targets(model, test_ds, max_batches=None)

    if use_vst:
        Y_true_o = inv_anscombe_tf(tf.convert_to_tensor(Y_true))
        Y_pred_o = inv_anscombe_tf(tf.convert_to_tensor(Y_pred))
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
    r2   = float(r2_score(yt, yp))

    psnr = float(tf.image.psnr(Y_true_m, Y_pred_m, max_val=1.0).numpy().mean())
    Y_true_2d = Y_true_m[:, Y_true_m.shape[1] // 2, :, :, :]
    Y_pred_2d = Y_pred_m[:, Y_pred_m.shape[1] // 2, :, :, :]
    ssim = float(tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=1.0).numpy().mean())

    print("\n=== Evaluation auf Test Set (Originalraum) ===")
    print(f"Modell       : {model_path.name}")
    print(f"INPUT_SHAPE  : {input_shape}")
    print(f"use_vst      : {use_vst}")
    print(f"MSE          : {mse:.6f}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"R2           : {r2:.6f}")
    print(f"PSNR         : {psnr:.2f} dB")
    print(f"SSIM (mid-Z) : {ssim:.4f}")

    # ---- Ergebnisse abspeichern ----
    results = {
        "model": model_path.name,
        "input_shape": tuple(int(x) for x in input_shape),
        "use_vst": use_vst,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "psnr": psnr,
        "ssim_midZ": ssim,
    }
    save_results(model_path, results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")


