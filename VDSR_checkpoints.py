# VDSR_checkpoints.py
from __future__ import annotations
import io, json
from pathlib import Path
import numpy as np
import tensorflow as tf



def get_out_dirs():
    root = Path.home() / "data" / "checkpoints_VDSR"
    return root

def make_epoch_ckpt_callback(run_name: str):
    """Speichert nur das beste Modell"""
    root = get_out_dirs()
    pattern = str(root / f"{run_name}__best.keras")
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=pattern,
        save_weights_only=False,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=0,
    )

def _model_summary_str(model: tf.keras.Model) -> str:
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    return s.getvalue()

def finalize_run(model, history, run_name, meta):
    root = get_out_dirs()

    # Beste Epoche bestimmen
    hist = history.history
    val = np.array(hist["val_loss"], dtype=float)
    trn = np.array(hist["loss"],     dtype=float)
    best_idx   = int(np.argmin(val))
    best_epoch = best_idx + 1
    best_trn   = float(trn[best_idx])
    best_val   = float(val[best_idx])

    # Weitere Metriken (optional vorhanden)
    def _pick(name: str):
        arr = hist.get(name)
        return float(arr[best_idx]) if arr is not None else None
    best_val_mse = _pick("val_mse")
    best_val_psnr = None
    for k in ("val_psnr", "val_psnr_metric"):
        v = hist.get(k)
        if v is not None:
            best_val_psnr = float(v[best_idx]); break

    best_path = root / f"{run_name}__best.keras"

    payload = dict(meta or {})
    payload.update({
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_loss_train": best_trn,
        "best_loss_val": best_val,
        "best_val_mse": best_val_mse,
        "best_val_psnr": best_val_psnr,
        "model_config": model.to_json(),
        "optimizer_config": model.optimizer.get_config() if hasattr(model, "optimizer") else None,
        "model_summary": _model_summary_str(model),
        "keras_path": str(best_path),
        "history": {k: [float(x) for x in v] for k, v in hist.items()},
    })

    # NumPy → JSON-safe
    def _to_jsonable(x):
        if isinstance(x, (np.floating,)):  return float(x)
        if isinstance(x, (np.integer,)):   return int(x)
        if isinstance(x, (np.ndarray,)):   return x.tolist()
        return x
    def _sanitize(obj):
        if isinstance(obj, dict):  return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_sanitize(v) for v in obj]
        return _to_jsonable(obj)

    with open(root / f"{run_name}__best.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2)

    return str(best_path)

def make_meta_dict(script_name: str,
                   batch_size: int,
                   epochs: int,
                   optimizer: tf.keras.optimizers.Optimizer,
                   learning_rate: float | None,
                   input_shape: tuple[int, int, int],
                   scale_range_train=(5000,15000),
                   scale_range_val=(10000,10001),
                   extra: dict | None = None):
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
    if extra: m.update(extra)
    return m