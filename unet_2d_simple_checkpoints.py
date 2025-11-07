# unet_2d_simple_checkpoints.py
from __future__ import annotations
import os, io, json, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorboard.plugins.hparams import api as hp


def _tb_root() -> Path:
    root = Path.home() / "data" / "tblogs_unet_2d_simple"
    root.mkdir(parents=True, exist_ok=True)
    return root

def make_run_dir(run_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = _tb_root() / f"{run_name}__{ts}"
    rd.mkdir(parents=True, exist_ok=False)
    return rd

def log_hparams_tb(log_dir: Path, hparams: dict):
    # file + TB HParams
    (log_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))
    with tf.summary.create_file_writer(str(log_dir)).as_default():
        hp.hparams({hp.HParam(k): v for k, v in hparams.items()})

class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.fw = tf.summary.create_file_writer(str(log_dir / "scalars"))
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        with self.fw.as_default():
            tf.summary.scalar("learning_rate", lr, step=epoch)

class ImageLogger(tf.keras.callbacks.Callback):
    """
    Loggt Eingabe/Ziel/Prediction als Bilder.
    Erwartet (x_vis, y_vis) mit gleicher Preprocessing-Pipeline wie Training/Val.
    Für 2D: (N,H,W,1) oder (N,H,W,3).
    Für 3D würde man Slices loggen – hier 2D-Version.
    """
    def __init__(self, log_dir: Path, sample_batch, tag_prefix="val", every_n_epochs=1, max_outputs=3):
        super().__init__()
        self.x, self.y = sample_batch
        self.every = max(1, int(every_n_epochs))
        self.max_outputs = max_outputs
        self.fw = tf.summary.create_file_writer(str(log_dir / "images"))
        # in [0,1] clampen, damit tf.summary.image nicht meckert
        self.x = tf.clip_by_value(tf.cast(self.x, tf.float32), 0.0, 1.0)
        self.y = tf.clip_by_value(tf.cast(self.y, tf.float32), 0.0, 1.0)
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0: 
            return
        p = self.model.predict(self.x, verbose=0)
        p = tf.clip_by_value(tf.cast(p, tf.float32), 0.0, 1.0)
        with self.fw.as_default():
            tf.summary.image(f"{self.tag_prefix}/x",  self.x, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag_prefix}/y",  self.y, step=epoch, max_outputs=self.max_outputs)
            tf.summary.image(f"{self.tag_prefix}/p",  p,      step=epoch, max_outputs=self.max_outputs)

def make_tb_callbacks(run_dir: Path, *, histograms=False, profile=False,
                      image_sample=None, image_every=1) -> list[tf.keras.callbacks.Callback]:
    histogram_freq = 1 if histograms else 0
    profile_batch = (50, 60) if profile else 0
    cbs = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(run_dir),
            histogram_freq=histogram_freq,
            write_graph=True,
            write_images=False,   # Bilder loggen wir gezielt via ImageLogger
            update_freq="epoch",
            profile_batch=profile_batch,
        ),
        LRLogger(run_dir),
    ]
    if image_sample is not None:
        cbs.append(ImageLogger(run_dir, image_sample, tag_prefix="val", every_n_epochs=image_every, max_outputs=3))
    return cbs


# Zielverzeichnis für Modelle/JSONs
def get_out_dirs() -> Path:
    root = Path.home() / "data" / "checkpoints_unet_2d_simple"
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
