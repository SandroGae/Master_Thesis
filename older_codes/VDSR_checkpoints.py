# VDSR_checkpoints.py
from __future__ import annotations
import os, io, json, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf


# Zielverzeichnis fuer Modelle/JSONs
def get_out_dirs() -> Path:
    root = Path.home() / "data" / "checkpoints_VDSR"
    root.mkdir(parents=True, exist_ok=True)
    return root


# Speichert während des Trainings immer nur das beste Modell (nach val_loss)
def make_epoch_ckpt_callback(run_name: str) -> tf.keras.callbacks.ModelCheckpoint:
    root = get_out_dirs()
    pattern = str(root / f"{run_name}__best.keras")  # fester Temp-Name
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=pattern,
        save_weights_only=False,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,  # zeigt "Saving model to ..." an
    )


def _model_summary_str(model: tf.keras.Model) -> str:
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    return s.getvalue()


def finalize_run(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    run_name: str,
    meta: dict | None,
    store_arch: str = "summary",   # "summary" | "json_file" | "none"
) -> str:
    """
    Speichert das beste Keras-File dauerhaft mit Loss im Dateinamen und legt
    ein schlankes JSON ab. 'model_config' wird NICHT in die JSON gepackt.

    store_arch:
      - "summary": nur 'model_summary' ins JSON (lesbar) [Default]
      - "json_file": Architektur separat als *.model.json (eingezogen)
      - "none": weder Summary noch Architektur (nicht empfohlen)
    """
    root = get_out_dirs()

    # --- beste Epoche / Kennzahlen
    hist = history.history
    val = np.array(hist["val_loss"], dtype=float)
    trn = np.array(hist["loss"],     dtype=float)
    best_idx   = int(np.argmin(val))
    best_epoch = best_idx + 1
    best_trn   = float(trn[best_idx])
    best_val   = float(val[best_idx])

    def _pick(name: str):
        arr = hist.get(name)
        return float(arr[best_idx]) if arr is not None else None

    best_val_mse  = _pick("val_mse")
    best_val_psnr = None
    for k in ("val_psnr", "val_psnr_metric"):
        v = hist.get(k)
        if v is not None:
            best_val_psnr = float(v[best_idx]); break

    # --- Temp-Checkpoint (vom Callback) -> finaler Name mit Losses
    tmp_ckpt   = root / f"{run_name}__best.keras"
    base_name  = f"{run_name}_loss{best_trn:.4f}_val{best_val:.4f}"
    final_keras = root / f"{base_name}.keras"
    final_json  = root / f"{base_name}.json"

    # Falls der Callback nichts geschrieben hat (z.B. 1 Epoche, kein Update):
    if not tmp_ckpt.exists():
        model.save(tmp_ckpt)
    os.replace(tmp_ckpt, final_keras)

    # --- Architektur optional separat ablegen
    arch_path = None
    if store_arch == "json_file":
        arch_path = root / f"{base_name}.model.json"
        cfg = json.loads(model.to_json())  # to_json() -> str -> wieder dict
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=root, prefix=f"{run_name}_", suffix=".model.json"
        ) as tmpf:
            json.dump(cfg, tmpf, indent=2)
            tmp_arch = tmpf.name
        os.replace(tmp_arch, arch_path)

    # --- JSON-Payload (schlank, ohne model_config)
    payload = dict(meta or {})
    payload.update({
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_loss_train": best_trn,
        "best_loss_val": best_val,
        "best_val_mse": best_val_mse,
        "best_val_psnr": best_val_psnr,
        "optimizer_class": model.optimizer.__class__.__name__ if hasattr(model, "optimizer") else None,
        "learning_rate_effective": (
            float(tf.keras.backend.get_value(model.optimizer.learning_rate))
            if hasattr(model, "optimizer") else None
        ),
        "model_summary": _model_summary_str(model) if store_arch == "summary" else None,
        "keras_path": str(final_keras),
        "arch_file": str(arch_path) if arch_path is not None else None,
        "history": {k: [float(x) for x in v] for k, v in hist.items()},
    })

    # --- NumPy -> JSON-safe und atomar schreiben
    def _to_jsonable(x):
        if isinstance(x, (np.floating,)):  return float(x)
        if isinstance(x, (np.integer,)):   return int(x)
        if isinstance(x, (np.ndarray,)):   return x.tolist()
        return x

    def _sanitize(obj):
        if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_sanitize(v) for v in obj]
        return _to_jsonable(obj)

    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=root, prefix=f"{run_name}_", suffix=".json"
    ) as tmpf:
        json.dump(_sanitize(payload), tmpf, indent=2)
        tmp_json = tmpf.name
    os.replace(tmp_json, final_json)

    # --- CSV (falls vorhanden) gleich wie Keras/JSON umbenennen
    csv_tmp = root / f"{run_name}.csv"
    final_csv = root / f"{base_name}.csv"
    if csv_tmp.exists():
        os.replace(csv_tmp, final_csv)
                   
    return str(final_keras)


def make_meta_dict(
    script_name: str,
    batch_size: int,
    epochs: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float | None,
    input_shape: tuple[int, int, int],
    scale_range_train=(5000, 15000),
    scale_range_val=(10000, 10001),
    extra: dict | None = None,
) -> dict:
    m = {
        "script_name": script_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer_class": optimizer.__class__.__name__,
        "learning_rate": float(learning_rate) if learning_rate is not None else
                         float(tf.keras.backend.get_value(optimizer.learning_rate)),
        "input_shape": list(input_shape),
        "normalization": {
            "clip_min": 0.0,
            "sum_norm": True,
            "scale_range_train": list(scale_range_train),
            "scale_range_val":   list(scale_range_val),
            "final_clip_01": True,
        },
    }
    if extra:
        m.update(extra)
    return m
