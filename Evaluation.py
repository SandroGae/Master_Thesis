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
import os, re, sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import r2_score

# === Import deiner Daten-Pipeline ===
from unet_3d_data import prepare_in_memory_5to5

# ====== Utils: Anscombe (nur fuer Metriken im Originalraum) ======
def inv_anscombe_tf(z):
    z = tf.maximum(z, 1e-3)
    z2 = z / 2.0
    y = z2**2 - 1.0/8.0 + 1.0/(4.0*z**2) - 11.0/(8.0*z**4)
    return tf.nn.relu(y)

# ====== Auswahl: Checkpoint-Ordner + Modellversion ======
DATA_ROOT = Path.home() / "data"

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
            return cand[int(s)-1]

def pick_version(ckpt_dir: Path):
    models = []
    pat = re.compile(r"^V(\d+)_.*\.keras$")
    for p in ckpt_dir.iterdir():
        if p.is_file() and p.suffix == ".keras" and pat.match(p.name):
            models.append(p)
    if not models:
        print(f"Keine V*-Modelle in {ckpt_dir} gefunden.")
        sys.exit(1)
    # Nach V-Nummer sortieren
    models.sort(key=lambda p: int(p.stem.split('_')[0][1:]))  # V{n}_...
    print(f"Waehle Modell in {ckpt_dir.name}:")
    for i, p in enumerate(models, 1):
        print(f"  [{i}] {p.name}")
    while True:
        s = input("Nummer: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(models):
            return models[int(s)-1]

# ====== Test-Dataset bauen (identisch zur Trainings-Pipeline) ======
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

# ====== Predictions einsammeln ======
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

# ====== Hauptlogik ======
def main():
    ckpt_dir = pick_checkpoint_dir()
    model_path = pick_version(ckpt_dir)

    # Heuristik: wenn Ordnername "anscombe" enthaelt -> use_vst=True
    use_vst = ("anscombe" in ckpt_dir.name.lower())

    print(f"\n>> Lade Modell: {model_path}")
    # Robust laden: erklaere evtl. Custom-Objekte (egal was beim Training genutzt wurde)
    def dummy(*args, **kwargs):  # falls nicht gebraucht
        return None
    custom_objects = {
        "combined_loss": globals().get("combined_loss", dummy),
        "ms_ssim_loss": globals().get("ms_ssim_loss", dummy),
        "ms_ssim_metric": globals().get("ms_ssim_metric", dummy),
        "psnr_metric": globals().get("psnr_metric", dummy),
        "psnr_orig_metric": globals().get("psnr_orig_metric", dummy),
        "ms_ssim_orig_metric": globals().get("ms_ssim_orig_metric", dummy),
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    print(">> Baue Test-Dataset… (use_vst =", use_vst, ")")
    test_ds, input_shape = build_test_dataset(
        use_vst=use_vst, size=5, group_len=41, dtype=np.float32, batch_size=4
    )

    # Vorhersagen
    Y_true, Y_pred = collect_preds_and_targets(model, test_ds, max_batches=None)

    # Basis-Fehler (im Trainingsraum)
    yt = Y_true.ravel()
    yp = Y_pred.ravel()
    mse  = float(np.mean((yt - yp) ** 2))
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(yt, yp))

    # PSNR / SSIM: je nach use_vst im Originalraum
    if use_vst:
        Y_true_o = inv_anscombe_tf(tf.convert_to_tensor(Y_true))
        Y_pred_o = inv_anscombe_tf(tf.convert_to_tensor(Y_pred))
        # clamp in [0,1] fuer diese Metriken, wie in deinen Trainingsmetriken
        Y_true_o = tf.clip_by_value(Y_true_o, 0.0, 1.0)
        Y_pred_o = tf.clip_by_value(Y_pred_o, 0.0, 1.0)

        psnr = float(tf.image.psnr(Y_true_o, Y_pred_o, max_val=1.0).numpy().mean())

        Y_true_2d = Y_true_o[:, Y_true_o.shape[1] // 2, :, :, :]
        Y_pred_2d = Y_pred_o[:, Y_pred_o.shape[1] // 2, :, :, :]
        ssim = float(tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=1.0).numpy().mean())
    else:
        psnr = float(tf.image.psnr(Y_true, Y_pred, max_val=1.0).numpy().mean())
        Y_true_2d = Y_true[:, Y_true.shape[1] // 2, :, :, :]
        Y_pred_2d = Y_pred[:, Y_pred.shape[1] // 2, :, :, :]
        ssim = float(tf.image.ssim(Y_true_2d, Y_pred_2d, max_val=1.0).numpy().mean())

    print("\n=== Evaluation auf Test Set ===")
    print(f"Modell       : {model_path.name}")
    print(f"INPUT_SHAPE  : {input_shape}")
    print(f"use_vst      : {use_vst}")
    print(f"MSE          : {mse:.6f}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"R2           : {r2:.6f}")
    print(f"PSNR         : {psnr:.2f} dB")
    print(f"SSIM (mid-Z) : {ssim:.4f}")

if __name__ == "__main__":
    main()

