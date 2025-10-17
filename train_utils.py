# train_utils.py
import h5py
import re as _re
from pathlib import Path as _Path

import os, sys, inspect, json, socket, getpass, platform, subprocess, time, uuid
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import atexit, signal

# ---------- kleine Helfer ----------
def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S")

def _safe_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None

def _serialize_optimizer(opt):
    try:
        return tf.keras.optimizers.serialize(opt)
    except Exception:
        return None

def _auto_code_name() -> str:
    try:
        main_mod = sys.modules.get("__main__")
        if main_mod and getattr(main_mod, "__file__", None):
            return os.path.splitext(os.path.basename(main_mod.__file__))[0]
    except Exception:
        pass
    try:
        if sys.argv and sys.argv[0]:
            return os.path.splitext(os.path.basename(sys.argv[0]))[0]
    except Exception:
        pass
    try:
        for fr in inspect.stack():
            fn = fr.filename
            if fn and fn not in ("<stdin>", "<string>"):
                return os.path.splitext(os.path.basename(fn))[0]
    except Exception:
        pass
    for k in ("SLURM_JOB_NAME", "PBS_JOBNAME", "JOB_NAME"):
        v = os.environ.get(k)
        if v:
            return v
    return "MODEL"

def _safe_unlink(p: Path):
    try:
        if p and Path(p).exists():
            Path(p).unlink()
    except Exception:
        pass

def purge_stale_temps(ckpt_root: Path, code_prefix: str):
    """
    Entfernt alle *_TEMP_*.keras fuer den gegebenen Code-Prefix.
    Sinnvoll zu Beginn eines Trainingslaufs.
    """
    ckpt_root = Path(ckpt_root)
    for p in ckpt_root.glob(f"{code_prefix}_TEMP_*.keras"):
        _safe_unlink(p)

def register_temp_cleanup(tmp_path: Path):
    """
    Registriert Aufraeum-Routinen fuer Exit & Signale, damit TEMP bei Abbruch nicht liegen bleibt.
    """
    def _cleanup(*_args, **_kwargs):
        _safe_unlink(tmp_path)
    # bei normalem Prozessende
    atexit.register(_cleanup)
    # bei Ctrl+C / kill
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old = signal.getsignal(sig)
            def handler(sig_, frame):
                _cleanup()
                # alten Handler (falls vorhanden) weiter aufrufen
                if callable(old):
                    old(sig_, frame)
            signal.signal(sig, handler)
        except Exception:
            # z.B. in manchen Umgebungen nicht erlaubt – ignorieren
            pass

# ---------- Logging ----------
class CompactLogger(callbacks.Callback):
    """
    Einheitlicher, kompakter Epochen-Logger.
    Gibt nur Keys aus, die wirklich im logs-Dict vorhanden sind.
    """
    def __init__(self, cols=None, show_time=True):
        super().__init__()
        self.cols = cols or [
            "loss","val_loss","mae","val_mae","mse","val_mse",
            "ssim","val_ssim",               # falls SSIM-Metrik genutzt wird
            "ms_ssim_metric","val_ms_ssim_metric",  # falls MS-SSIM-Metrik genutzt wird
            "psnr","val_psnr"
        ]
        self.show_time = show_time
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.show_time:
            self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        dt = (time.time() - self._t0) if (self.show_time and self._t0 is not None) else None
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = None

        parts = [f"E{epoch+1:03d}"]
        for k in self.cols:
            v = logs.get(k, None)
            # np.isfinite akzeptiert auch Skalar/np.float32
            if v is not None and np.all(np.isfinite(v)):
                parts.append(f"{k}={v:7.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.1e}")
        if dt is not None:
            parts.append(f"time={dt:5.1f}s")
        print(" | ".join(parts))

# ---------- Guards ----------
class LossNaNGuard(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        loss = (logs or {}).get("loss", None)
        if loss is not None and not np.isfinite(loss):
            print(f"[NaNLoss] batch={batch} loss={loss}")
            self.model.stop_training = True

class WeightNaNGuard(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        for w in self.model.weights:
            if not tf.reduce_all(tf.math.is_finite(w)):
                print(f"[NaNWeight] layer={w.name} batch={batch}")
                self.model.stop_training = True
                break

# ---------- Naming / Ranking Callback ----------
class BestFinalizeCallback(callbacks.Callback):
    """
    Am Ende des Trainings:
      1) TEMP-Checkpoint von ModelCheckpoint nach <code>_NEW_valloss_...>.keras verschieben
      2) Begleit-JSON mit Metadaten schreiben
      3) Alle Modelle gleichen Prefix nach val_loss ranken: <code>_V1_..., _V2_..., ...
    """
    def __init__(self, root: Path, run_meta: dict = None, tmp_name: str = None, code_name: str = "AUTO"):
        super().__init__()
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        auto = _auto_code_name() if (code_name is None or str(code_name).upper() == "AUTO") else code_name
        self.code = self._sanitize_code(auto)
        temp_dir = self.root / "TEMP"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path = temp_dir / (tmp_name or f"{self.code}_TEMP_{uuid.uuid4().hex}.keras")
        self.best_val_loss = np.inf
        self.best_psnr = None
        self.run_meta = run_meta or {}

    @staticmethod
    def _sanitize_code(code: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (code or "").strip())
        return safe or "MODEL"

    def on_epoch_end(self, epoch, logs=None):
        if not logs or "val_loss" not in logs:
            return
        vloss = float(logs["val_loss"])
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            # PSNR kann als "psnr" oder "psnr_metric" auftauchen – beide prüfen
            psnr = logs.get("psnr")
            if psnr is None:
                psnr = logs.get("psnr_metric")
            self.best_psnr = float(psnr) if psnr is not None else None

    def on_train_end(self, logs=None):
        vloss_str = f"{self.best_val_loss:.3e}" if np.isfinite(self.best_val_loss) else "nan"
        psnr_part = f"_PSNR_{self.best_psnr:.3g}" if (self.best_psnr is not None and np.isfinite(self.best_psnr)) else ""
        new_model = self.root / f"{self.code}_valloss_{vloss_str}{psnr_part}.keras"
        if self.tmp_path.exists() and self.tmp_path.stat().st_size > 0:
            try:
                os.replace(self.tmp_path, new_model)
            except Exception as e:
                print(f"[WARN] Konnte TEMP nicht nach NEW umbenennen: {e}")
                return
            self._write_json_for_model(new_model)
        self._rank_all_models()
        try:
            if self.tmp_path.exists():
                os.remove(self.tmp_path)
        except Exception:
            pass

    def _write_json_for_model(self, model_path: Path):
        json_path = model_path.with_suffix(".json")
        try:
            inp_shape = tuple(int(x) for x in (self.model.input_shape or []) if isinstance(x, (int, np.integer)))
        except Exception:
            inp_shape = None
        try:
            loss_name = getattr(self.model.loss, "__name__", str(self.model.loss))
        except Exception:
            loss_name = None
        try:
            metrics_list = [getattr(m, "__name__", str(m)) for m in (self.model.metrics or [])]
        except Exception:
            metrics_list = None
        try:
            arch_json = self.model.to_json()
        except Exception:
            arch_json = None

        meta = {
            "timestamp": _timestamp(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_commit": _safe_git_commit(),
            "code_name": self.code,
            # alles was du bei build_standard_callbacks reinreichst:
            "batch_size": self.run_meta.get("batch_size"),
            "epochs_planned": self.run_meta.get("epochs"),
            "early_stopping": self.run_meta.get("early_stopping"),
            "data_prep": self.run_meta.get("data_prep"),
            "alpha": self.run_meta.get("alpha"),
            "loss_components": self.run_meta.get("loss_components"),
            "best_val_loss": float(self.best_val_loss) if np.isfinite(self.best_val_loss) else None,
            "best_psnr_metric": self.best_psnr,
            "input_shape": inp_shape,
            "loss": loss_name,
            "metrics": metrics_list,
            "optimizer": _serialize_optimizer(getattr(self.model, "optimizer", None)),
            "architecture_json": arch_json,
        }
        try:
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] Konnte JSON nicht schreiben: {e}")

    @staticmethod
    def _parse_filename_simple(name: str):
        if not name.endswith(".keras"):
            return None
        base = name[:-6]; parts = base.split("_")
        try: i_vl = parts.index("valloss")
        except ValueError: return None
        if i_vl + 1 >= len(parts): return None
        try: val_loss = float(parts[i_vl + 1])
        except Exception: return None
        psnr = None
        try:
            i_ps = parts.index("PSNR")
            if i_ps + 1 < len(parts): psnr = float(parts[i_ps + 1])
        except ValueError:
            pass
        except Exception:
            psnr = None
        return {"val_loss": val_loss, "psnr": psnr}

    def _rank_all_models(self):
        items = []
        for p in self.root.glob(f"{self.code}_*.keras"):
            if not p.is_file(): continue
            meta = self._parse_filename_simple(p.name)
            if meta: items.append((p, meta["val_loss"], meta["psnr"]))
        if not items: return
        items.sort(key=lambda x: (x[1], x[0].stat().st_mtime))

        temps = []
        for path, vloss, psnr in items:
            base_stem = path.with_suffix("").name
            jsons = []
            p0 = self.root / (base_stem + ".json")
            if p0.exists(): jsons.append(p0)
            t_model = self.root / f".tmp_{uuid.uuid4().hex}.keras"
            os.replace(path, t_model)
            tmp_jsons = []
            for j in jsons:
                ts_suffix = j.name[len(base_stem):]
                t_json = t_model.with_suffix("")
                t_json = t_json.parent / (t_json.name + ts_suffix)
                os.replace(j, t_json)
                tmp_jsons.append((t_json, ts_suffix))
            temps.append((t_model, tmp_jsons, vloss, psnr))

        for rank, (t_model, tmp_jsons, vloss, psnr) in enumerate(temps, start=1):
            v = f"{vloss:.3e}"
            ps = f"_PSNR_{psnr:.3g}" if psnr is not None else ""
            final_model = self.root / f"{self.code}_V{rank}_valloss_{v}{ps}.keras"
            os.replace(t_model, final_model)
            final_stem = final_model.with_suffix("").name
            for t_json, ts_suffix in tmp_jsons:
                if t_json.exists():
                    final_json = final_model.with_suffix("")
                    final_json = final_json.parent / (final_stem + ts_suffix)
                    os.replace(t_json, final_json)

# ---------- Bequeme Callback-Factory ----------
def build_standard_callbacks(
    ckpt_root: Path,
    run_meta: dict,
    *,
    monitor: str = "val_loss",
    patience_es: int = 20,
    reduce_on_plateau: bool = True,
    reduce_factor: float = 0.5,
    reduce_patience: int = 10,
    min_lr: float = 1e-6,
    include_nan_guards: bool = True, # Training bricht ab sobald NaN in Loss oder weights auftaucht
    include_logger: bool = True,     # Besseres Log als verbose
    code_name: str = "AUTO",
    verbose_ckpt: int = 1
):
    # --- NEU: vor dem Training alte TEMP-Files fuer diesen Prefix loeschen
    prefix = _auto_code_name() if (code_name is None or str(code_name).upper() == "AUTO") else code_name
    prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (prefix or "").strip()) or "MODEL"
    purge_stale_temps(ckpt_root, prefix)

    bf = BestFinalizeCallback(ckpt_root, run_meta=run_meta, code_name=code_name)

    # --- NEU: falls der Lauf abgebrochen wird, TEMP-Datei aufraeumen
    register_temp_cleanup(bf.tmp_path)

    ckpt_best = callbacks.ModelCheckpoint(
        filepath=str(bf.tmp_path),
        monitor=monitor, mode="min", save_best_only=True, verbose=verbose_ckpt,
    )

    cb_list = []
    if include_nan_guards:
        cb_list += [WeightNaNGuard(), LossNaNGuard(), callbacks.TerminateOnNaN()]
    if reduce_on_plateau:
        cb_list += [callbacks.ReduceLROnPlateau(monitor=monitor, factor=reduce_factor,
                                                patience=reduce_patience, min_lr=min_lr, verbose=0)]
    cb_list += [callbacks.EarlyStopping(monitor=monitor, patience=patience_es,
                                        restore_best_weights=True, verbose=0)]
    cb_list += [ckpt_best, bf]
    if include_logger:
        cb_list += [CompactLogger()]

    return cb_list, bf, ckpt_best


# Optional: universeller Clamper (nützlich für Loss/Metric-Helfer)
def clip01(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)

class H5FiveStackGrouped:
    """
    Liefert 5er-Stacks (D=5,H,W,1) aus HDF5 mit stride=1 *innerhalb* von 41er-Gruppen.
    high=/high_count/data, low=/low_count/data (Shape: H,W,N).
    Kein Crossing über Gruppenränder. Pro Gruppe entstehen (group_len - 5 + 1) Samples.
    """
    def __init__(self, path: _Path, group_len=41, stack_depth=5, dtype=np.float32):
        assert int(stack_depth) == 5, "Diese Klasse ist auf D=5 ausgelegt."
        self.f = h5py.File(str(path), "r")
        self.high = self.f["/high_count/data"]  # (H, W, N)
        self.low  = self.f["/low_count/data"]   # (H, W, N)
        self.H, self.W, self.N = self.high.shape
        self.group_len = int(group_len)
        self.D = 5
        self.dtype = dtype

        if self.N % self.group_len != 0:
            raise ValueError(f"N={self.N} ist kein Vielfaches von group_len={self.group_len}")
        self.num_groups = self.N // self.group_len
        self.samples_per_group = self.group_len - self.D + 1  # 37 bei 41/5
        self.total_samples = self.num_groups * self.samples_per_group

    def __len__(self): return self.total_samples

    def _flat_to_group_offset(self, k: int) -> int:
        g = k // self.samples_per_group
        o = k %  self.samples_per_group
        return g * self.group_len + o

    def get_pair(self, k: int):
        start = self._flat_to_group_offset(k)
        idxs = np.arange(start, start + self.D)
        hi = np.asarray(self.high[..., idxs], dtype=self.dtype)  # (H,W,5)
        lo = np.asarray(self.low[...,  idxs], dtype=self.dtype)
        # → (5,H,W,1)
        hi = np.moveaxis(hi, -1, 0)[..., None]
        lo = np.moveaxis(lo, -1, 0)[..., None]
        return lo, hi


def make_stream_ds_grouped(h5obj: H5FiveStackGrouped, *,
                           batch_size=32,
                           shuffle=False,
                           preproc=None,
                           augmenter=None,
                           prefetch=tf.data.AUTOTUNE,
                           shuffle_buffer=1024,
                           deterministic=False):
    """
    Baut tf.data-Dataset aus H5FiveStackGrouped.
    Reihenfolge: generator → cast → preproc (z.B. Normalisierung) → augment → batch → prefetch.
    preproc/augmenter sind Callables (x,y)→(x',y').
    """
    def gen():
        for k in range(len(h5obj)):
            x, y = h5obj.get_pair(k)
            yield x, y

    out_spec = (
        tf.TensorSpec(shape=(h5obj.D, h5obj.H, h5obj.W, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h5obj.D, h5obj.H, h5obj.W, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_spec)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)

    if preproc is not None:
        ds = ds.map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
    if augmenter is not None:
        ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(int(shuffle_buffer), len(h5obj)),
                        reshuffle_each_iteration=True)

    ds = ds.batch(int(batch_size)).prefetch(prefetch)
    if not deterministic:
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)
    return ds


def build_5stack_datasets(train_path, val_path, test_path, *,
                          group_len=41,
                          batch_train=32, batch_eval=32,
                          preproc_train=None, preproc_eval=None,
                          augmenter=None,
                          deterministic=False,
                          prefetch=tf.data.AUTOTUNE,
                          shuffle_buffer=1024):
    """
    Komfort-Builder: öffnet drei HDF5s, erzeugt je ein Dataset.
    Normalisierung bleibt *außen* (z.B. aus jens_stuff.SumScaleNormalizer via Wrapper).
    """
    train_h5 = H5FiveStackGrouped(_Path(train_path), group_len=group_len, stack_depth=5)
    val_h5   = H5FiveStackGrouped(_Path(val_path),   group_len=group_len, stack_depth=5)
    test_h5  = H5FiveStackGrouped(_Path(test_path),  group_len=group_len, stack_depth=5)

    input_shape = (5, train_h5.H, train_h5.W, 1)

    train_ds = make_stream_ds_grouped(
        train_h5, batch_size=batch_train, shuffle=True,
        preproc=preproc_train, augmenter=augmenter,
        prefetch=prefetch, shuffle_buffer=shuffle_buffer, deterministic=deterministic
    )
    val_ds = make_stream_ds_grouped(
        val_h5, batch_size=batch_eval, shuffle=False,
        preproc=preproc_eval, prefetch=prefetch, deterministic=deterministic
    )
    test_ds = make_stream_ds_grouped(
        test_h5, batch_size=batch_eval, shuffle=False,
        preproc=preproc_eval, prefetch=prefetch, deterministic=deterministic
    )

    meta = {
        "H": train_h5.H, "W": train_h5.W, "D": 5,
        "group_len": group_len,
        "samples_per_group": group_len - 5 + 1,
        "input_shape": input_shape,
        "train_total_samples": len(train_h5),
        "val_total_samples": len(val_h5),
        "test_total_samples": len(test_h5),
    }
    return train_ds, val_ds, test_ds, meta