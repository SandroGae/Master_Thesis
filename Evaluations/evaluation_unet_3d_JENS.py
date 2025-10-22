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

from jens_stuff import SumScaleNormalizer
from unet_3d_data_JENS import prepare_in_memory_5to5

DATA_ROOT = Path.home() / "data"
EVAL_ROOT = DATA_ROOT  # Verzeichnis für Ausgabe

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
    yt = tf.convert_to_tensor(Y_true, dtype=tf.float32)
    yp = tf.convert_to_tensor(Y_pred, dtype=tf.float32)
    yt2 = tf.reshape(yt, (-1, yt.shape[2], yt.shape[3], yt.shape[4]))
    yp2 = tf.reshape(yp, (-1, yp.shape[2], yp.shape[3], yp.shape[4]))
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
def map_slice_wise(normalizer):
    def _finite01(t):
        t = tf.cast(t, tf.float32)
        t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
        return tf.clip_by_value(t, 0.0, 1.0)
    def _fn(x, y):
        x_norm, y_norm = normalizer.map(x, y)
        return _finite01(x_norm), _finite01(y_norm)
    return _fn

# <<< HIER global (vor build_test_dataset) definieren
preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001],
    pre_offset=0.0,
    normalize_label=True,
    batch_mode=True,
    axis=(1, 2, 3),
    clip_before=[0., float("inf")],
    clip_after=[0., 1.],
)

def build_test_dataset(size=5, group_len=41, dtype=np.float32, batch_size=4):
    results = prepare_in_memory_5to5(
        data_dir=Path.home() / "data" / "original_data",
        size=size, group_len=group_len, dtype=dtype,
    )
    X_test, Y_test = results["test"]

    ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    ds = ds.map(lambda x, y: tuple(map_slice_wise(preproc_valid_slice)(x, y)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, X_test.shape[1:]


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

# ========== Erstelle File Namen ==========
def _build_eval_filename(model_path: Path, mae_value: float, psnr_value: float, prefix: Optional[str] = None) -> str:
    import re
    stem = model_path.stem

    # 1) Trainings-Infos aus dem Modell-Stem entfernen
    #    - PSNR Teil (Training)
    stem_clean = re.sub(r"_PSNR_([0-9.]+)", "", stem, flags=re.I)
    #    - val / valloss in ALLEN Schreibweisen: _val..., _valloss..., optional '=' oder '_' davor,
    #      Zahl auch in wissenschaftlicher Notation (4.104e-02)
    stem_clean = re.sub(r"_val(?:_?loss)?(?:=|_)?[0-9.eE+\-]+", "", stem_clean, flags=re.I)

    # 2) Prefix bestimmen und "evaluation_" abstreifen, um Doppelungen zu vermeiden
    pref_full = _sanitize_name(prefix) if prefix else _auto_script_name()
    model_pref = re.sub(r"^evaluation[_-]?", "", pref_full, flags=re.I)  # z.B. "evaluation_unet_3d" -> "unet_3d"

    # 3) Prefix nur voranstellen, wenn der Name nicht ohnehin so beginnt
    stem_l = stem_clean.lower()
    if stem_l.startswith(model_pref.lower() + "_") or stem_l.startswith(pref_full.lower() + "_"):
        name = stem_clean
    else:
        name = f"{model_pref}_{stem_clean}"

    # 4) Nur TEST-Kennzahlen anhängen (keine Trainingswerte mehr)
    return f"{name}_mae{mae_value:.6f}_psnr{psnr_value:.2f}.json"





# ========== Ergebnisse speichern ==========
def save_results(model_path: Path, results: dict):
    out_dir = EVAL_ROOT / "model_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    mae_value  = float(results.get("mae", 0.0))
    psnr_value = float(results.get("psnr", 0.0))

    out_name = _build_eval_filename(model_path, mae_value, psnr_value)  # <— nur Testwerte
    out_path = out_dir / out_name

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n>> Ergebnisse gespeichert unter: {out_path}")


# ========== Main ==========
def main():
    ckpt_dir = pick_checkpoint_dir()
    model_path = pick_version(ckpt_dir)

    print(f"\n>> Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode = False)

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

