# %%
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





def build_VDSR_datasets(
    data_dir,
    batch_train,
    batch_eval,
    read_block,
    preproc_train,
    preproc_eval,
    cache_after_preproc,
):
    """
    Liest 2D-Bilder (H,W,N) aus den HDF5-Dateien:
        /high_count/data und /low_count/data
    und baut daraus tf.data.Datasets für Training, Validation und Test.
    Rückgabe: (train_ds, val_ds, test_ds, meta)
    """

    # Hilfsfunktion: Blockweises Laden für Performance!
    def read_hdf5_block(h5file, start, count, dtype=np.float32):
        end = min(start + count, h5file["/high_count/data"].shape[2])
        sl = slice(start, end)
        high = np.asarray(h5file["/high_count/data"][..., sl], dtype=dtype)
        low  = np.asarray(h5file["/low_count/data"][...,  sl], dtype=dtype)

        print(f"[read_hdf5_block] Roh-Shape high={high.shape}, low={low.shape}")

        # (H,W,M) → (M,H,W,1)
        high = np.moveaxis(high, 2, 0)[..., None]
        low  = np.moveaxis(low,  2, 0)[..., None]

        print(f"[read_hdf5_block] Nach Rearrange high={high.shape}, low={low.shape}")
        return low, high


    # Öffne Dateien
    tr_file = h5py.File(str(data_dir / "training_data.hdf5"), "r")
    va_file = h5py.File(str(data_dir / "validation_data.hdf5"), "r")
    te_file = h5py.File(str(data_dir / "test_data.hdf5"), "r")

    H, W, Ntr = tr_file["/high_count/data"].shape
    Nva = va_file["/high_count/data"].shape[2]
    Nte = te_file["/high_count/data"].shape[2]

    print(f"[build_VDSR_datasets] Shapes: train={Ntr}, val={Nva}, test={Nte}  |  Image size: ({H}x{W})")

    # ==========================
    # Hilfsfunktion für Dataset-Erzeugung
    # ==========================
    def make_dataset(h5file, n_images, batch_size, shuffle, preproc_fn):
        starts = tf.data.Dataset.range(0, n_images, read_block)
        if shuffle:
            starts = starts.shuffle(buffer_size=max(1, n_images // read_block),
                                    reshuffle_each_iteration=True)
        print(f"[make_dataset] n_images={n_images}, batch_size={batch_size}, read_block={read_block}, shuffle={shuffle}")

        def load_block_py(start_idx):
            start_idx = int(start_idx)
            print(f"[make_dataset -> load_block_py] Lade Block ab Index {start_idx}")
            low, high = read_hdf5_block(h5file, start_idx, read_block)
            print(f"[make_dataset -> load_block_py] Rückgabe-Shape low={low.shape}, high={high.shape}")
            return low, high  # numpy arrays

        ds = starts.map(
            lambda s: tf.numpy_function(load_block_py, [s], Tout=(tf.float32, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.unbatch()  # (M,H,W,1)
        print(f"[make_dataset] Nach unbatch → einzelne Samples (H,W,1)")

        if shuffle:
            ds = ds.shuffle(buffer_size=n_images, reshuffle_each_iteration=True)
            print(f"[make_dataset] Shuffle aktiviert (buffer={n_images})")

        # Preprocessing
        ds = ds.map(preproc_fn, num_parallel_calls=tf.data.AUTOTUNE)
        print("[make_dataset] Preprocessing-Funktion angewendet")

        if cache_after_preproc:
            ds = ds.cache()
            print("[make_dataset] Dataset nach Preprocessing gecached")

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        print(f"[make_dataset] Dataset finalisiert mit Batching + Prefetching\n")
        return ds

    # ==========================
    # Baue train / val / test
    # ==========================
    print("[build_VDSR_datasets] Erzeuge train_ds ...")
    train_ds = make_dataset(tr_file, Ntr, batch_train, shuffle=True,  preproc_fn=preproc_train)
    print("[build_VDSR_datasets] Erzeuge val_ds ...")
    val_ds   = make_dataset(va_file, Nva, batch_eval, shuffle=True,  preproc_fn=preproc_eval)
    print("[build_VDSR_datasets] Erzeuge test_ds ...")
    test_ds  = make_dataset(te_file, Nte, batch_eval, shuffle=False, preproc_fn=preproc_eval)

    # ==========================
    # Meta-Daten
    # ==========================
    input_shape = (H, W, 1)
    meta = {
        "H": H, "W": W, "D": 1,
        "input_shape": input_shape,
        "n_train": Ntr, "n_val": Nva, "n_test": Nte,
    }

    # Quick sanity-check: nimm eine Batchprobe und drucke Shapes
    print("[build_VDSR_datasets] Prüfe Beispielbatch ...")
    for xb, yb in train_ds.take(1):
        print(f"[Sample check] x_batch shape: {xb.shape}, y_batch shape: {yb.shape}")
        print(f"[Sample check] dtype x={xb.dtype}, y={yb.dtype}")
        break

    print("[build_VDSR_datasets] Finished building datasets.")
    return train_ds, val_ds, test_ds, meta









# Flat-Loader: Einzelbilder (D=1)
# -------------------------------
class H5Flat:
    """
    Erwartete Datasets in der HDF5:
        /high_count/data  -> Shape (H, W, N)
        /low_count/data   -> Shape (H, W, N)
    Liest blockweise (wird nicht alles in die RAM laden)
    """
    def __init__(self, path: Path, dtype=np.float32):
        self.file = h5py.File(str(path), "r")
        self.high = self.file["/high_count/data"]  # (H, W, N)
        self.low  = self.file["/low_count/data"]   # (H, W, N)
        H, W, N = self.high.shape
        self.H, self.W, self.N = int(H), int(W), int(N)
        self.dtype = dtype

    def get_block(self, start: int, count: int):
        """
        Liest mehrere aufeinanderfolgende Frames (war Bottleneck)
        Rückgabe (low, high) mit Shape (M, 1, H, W, 1), M = Anzahl Frames.
        """
        end = min(start + count, self.N)
        frame_slice = slice(start, end)

        high_raw = np.asarray(self.high[..., frame_slice], dtype=self.dtype)  # (H,W,M)
        low_raw  = np.asarray(self.low[...,  frame_slice], dtype=self.dtype)  # (H,W,M)

        # (H,W,M) -> (M,H,W)
        high_reo = np.moveaxis(high_raw, 2, 0)
        low_reo  = np.moveaxis(low_raw,  2, 0)

        # (M,H,W) -> (M,1,H,W,1)   (D=1, Channel=1)
        high_final = high_reo[:, None, ..., None]
        low_final  = low_reo[:,  None, ..., None]
        return low_final, high_final

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


def build_1stack_datasets_flat(
    *,
    data_dir: Path,
    batch_train: int,
    batch_eval: int,
    read_block: int,
    preproc_train,
    preproc_eval,
    out_rank: int, # 5 für Conv3D, 4 für Conv2D!!!
    cache_after_preproc: bool = False,
):

    """
    Erstellt tf.data.Datasets (train, val, test)
    """
    # Dateien oeffnen
    tr = H5Flat(data_dir / "training_data.hdf5")
    va = H5Flat(data_dir / "validation_data.hdf5")
    te = H5Flat(data_dir / "test_data.hdf5")

    # Hilfsfunktion für einen Split
    def make_split(flat: H5Flat, batch_size: int, shuffle: bool):
        starts = tf.data.Dataset.range(0, flat.N, read_block)
        if shuffle:
            # Shufflet die Blockstarts
            starts = starts.shuffle(
                buffer_size=max(1, flat.N // read_block),
                reshuffle_each_iteration=True,
            )

        def read_block_py(start_idx):
            # Dient nur zur I/O effizienz
            start_idx = int(start_idx)
            return flat.get_block(start_idx, read_block)  # (M,1,H,W,1)

        ds = starts.map(
            lambda s: tf.numpy_function(
                read_block_py, [s], Tout=(tf.float32, tf.float32)
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Shape-Guards (noch 5D: (None,1,H,W,1))
        ds = ds.map(
            lambda x, y: (
                tf.ensure_shape(x, (None, 1, flat.H, flat.W, 1)),
                tf.ensure_shape(y, (None, 1, flat.H, flat.W, 1)),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # In Einzelsamples auflösen: (M,1,H,W,1) -> M * (1,H,W,1)
        ds = ds.unbatch()
        if shuffle:
            # Shufflet die Einzelbilder nochmals global
            ds = ds.shuffle(buffer_size=flat.N, reshuffle_each_iteration=True)

        # Preprocessing (Training vs. Eval)
        if (preproc_train is not None) and (preproc_eval is not None):
            fn = preproc_train if shuffle else preproc_eval
            ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Optional: nach Preproc cachen
        if cache_after_preproc:
            ds = ds.cache()

        # Falls 4D Tensoren (wie in VDSR) -> D-Achse (axis=1) entfernen -> (B, H, W, C =1)
        if out_rank == 4:
            ds = ds.map(
                lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)),
                num_parallel_calls=tf.data.AUTOTUNE,
    )


        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_split(tr, batch_train, shuffle=True)
    val_ds   = make_split(va, batch_eval,   shuffle=True)
    test_ds  = make_split(te, batch_eval,   shuffle=False)

    # Input-Shape fuer das jeweilige Modell
    input_shape = (1, tr.H, tr.W, 1) if out_rank == 5 else (tr.H, tr.W, 1)

    meta = {
        "H": tr.H, "W": tr.W, "D": 1,
        "input_shape": input_shape,
        "n_train": tr.N, "n_val": va.N, "n_test": te.N,
    }
    return train_ds, val_ds, test_ds, meta




# Clamper auf [0, 1]
def clip01(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)




# ---- gruppenweiser Reader + vektorisiertes Fenster-Building ----
class H5Groups:
    def __init__(self, path: Path, group_len=41, dtype=np.float32):
        import h5py, numpy as np
        self.f = h5py.File(str(path), "r")
        self.high = self.f["/high_count/data"]   # (H,W,N)
        self.low  = self.f["/low_count/data"]
        H, W, N = self.high.shape
        if N % group_len != 0:
            raise ValueError(f"N={N} kein Vielfaches von group_len={group_len}")
        self.H, self.W, self.N = H, W, N
        self.group_len = int(group_len)
        self.num_groups = N // group_len
        self.dtype = dtype

    def __len__(self): return self.num_groups

    def get_group_windows(self, group_idx: int, win: int = 5):
        start = group_idx * self.group_len
        end = start + self.group_len
        hi = np.asarray(self.high[..., start:end], dtype=self.dtype)  # (H,W,group_len)
        lo = np.asarray(self.low[...,  start:end], dtype=self.dtype)  # (H,W,group_len)
        # Alle Sliding-Window-Indizes: (group_len - win + 1, win)
        idx = np.arange(self.group_len - win + 1)[:, None] + np.arange(win)[None, :]
        # (H,W,37,5)
        hi5 = hi[..., idx]
        lo5 = lo[..., idx]
        # -> (37,5,H,W,1)
        hi5 = np.moveaxis(hi5, (2, 3, 0, 1), (0, 1, 2, 3))[..., None]
        lo5 = np.moveaxis(lo5, (2, 3, 0, 1), (0, 1, 2, 3))[..., None]
        return lo5, hi5


    def __del__(self):
        try: self.f.close()
        except: pass


def build_5stack_datasets_grouped(data_dir: Path, *,
                                  group_len=41,
                                  batch_train=32, batch_eval=32,
                                  preproc_train=None, preproc_eval=None,
                                  augmenter=None,
                                  deterministic=False,
                                  prefetch=tf.data.AUTOTUNE,
                                  cache_after_preproc=False):
    import tensorflow as tf
    train_g = H5Groups(data_dir / "training_data.hdf5", group_len=group_len)
    val_g   = H5Groups(data_dir / "validation_data.hdf5", group_len=group_len)
    test_g  = H5Groups(data_dir / "test_data.hdf5",      group_len=group_len)

    input_shape = (5, train_g.H, train_g.W, 1)

    def ds_from_groups(greader, batch_size, shuffle_groups):
        # 1) range über Gruppen
        ds = tf.data.Dataset.range(len(greader))
        if shuffle_groups:
            ds = ds.shuffle(buffer_size=len(greader), reshuffle_each_iteration=True)

        # 2) gruppenweise laden (NumPy), jeweils (37,5,H,W,1) zurückgeben
        def _py_group(i):
            import numpy as np
            lo5, hi5 = greader.get_group_windows(int(i))
            return lo5, hi5

        ds = ds.map(lambda i: tf.numpy_function(_py_group, [i],
                                                Tout=(tf.float32, tf.float32)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        # shapes setzen (Keras mag bekannte Ränge):
        ds = ds.map(lambda x,y: (tf.ensure_shape(x, (None,)+input_shape),
                                 tf.ensure_shape(y, (None,)+input_shape)),
                    num_parallel_calls=tf.data.AUTOTUNE)

        # 3) unbatch → einzelne Samples
        ds = ds.unbatch()

        # 4) Preproc/augment
        if preproc_train is not None and preproc_eval is not None:
            # Entscheide anhand shuffle_groups, ob Train- oder Eval-Preproc:
            preproc = preproc_train if shuffle_groups else preproc_eval
            ds = ds.map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
        if augmenter is not None and shuffle_groups:
            ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

        if cache_after_preproc:
            ds = ds.cache()

        # 5) Batch + Prefetch + (optional) auf GPU vorziehen
        ds = ds.batch(batch_size)
        ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0")).prefetch(1)
        return ds

    train_ds = ds_from_groups(train_g, batch_train, shuffle_groups=True)
    val_ds   = ds_from_groups(val_g,   batch_eval,   shuffle_groups=False)
    test_ds  = ds_from_groups(test_g,  batch_eval,   shuffle_groups=False)

    if not deterministic:
        opts = tf.data.Options(); opts.experimental_deterministic = False
        train_ds = train_ds.with_options(opts)
        val_ds   = val_ds.with_options(opts)
        test_ds  = test_ds.with_options(opts)

    meta = {
        "H": train_g.H, "W": train_g.W, "D": 5,
        "group_len": group_len,
        "samples_per_group": group_len - 5 + 1,
        "input_shape": input_shape,
        "train_groups": len(train_g), "val_groups": len(val_g), "test_groups": len(test_g),
    }
    return train_ds, val_ds, test_ds, meta








# =============================================
# Files benennen / ranken / Metadaten schreiben
# =============================================

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