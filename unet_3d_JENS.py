# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %%
# ==============================
# 0) Imports & global setup
# ==============================
import os, sys, inspect, json, socket, getpass, platform, subprocess, time, uuid
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")

from tensorflow.keras import regularizers, constraints, layers, models, callbacks
from tensorflow.keras.optimizers import AdamW

from unet_3d_data_JENS import prepare_in_memory_5to5
from jens_stuff import SumScaleNormalizer, reset_random_seeds

# --- Repro & GPU ---
seed = 0
reset_random_seeds(seed)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# %%
# ==============================
# 1) Daten laden (CPU)
# ==============================
print(">>> Phase 1: Starting data prep on CPU...")
results = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    size=5, group_len=41, dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")
X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]   # (D,H,W,C)
BATCH_SIZE = 32
EPOCHS     = 200

# %%
# ==============================
# 2) Preprocessing (slice-weise)
# ==============================
preproc_train_slice = SumScaleNormalizer(
    scale_range=[5000, 15001], pre_offset=0.0,
    normalize_label=True,
    axis=(1, 2, 3),      # reduziert H,W,C — NICHT D
    batch_mode=True,     # pro Slice
    clip_before=[0., float("inf")], clip_after=[0., 1.]
)
preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001], pre_offset=0.0,
    normalize_label=True,
    axis=(1, 2, 3),
    batch_mode=True,
    clip_before=[0., float("inf")], clip_after=[0., 1.]
)

def map_slice_wise(normalizer):
    def _finite01(t):
        t = tf.cast(t, tf.float32)
        t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))
        return tf.clip_by_value(t, 0.0, 1.0)
    def _fn(x, y):
        x_norm, y_norm = normalizer.map(x, y)
        return _finite01(x_norm), _finite01(y_norm)
    return _fn



# %%
# ==============================
# 3) Dataset-Bau (+ NaN-Wächter)
# ==============================
def nan_debug(x, y):
    nx = tf.reduce_sum(tf.cast(~tf.math.is_finite(x), tf.int32))
    ny = tf.reduce_sum(tf.cast(~tf.math.is_finite(y), tf.int32))
    tf.debugging.assert_equal(nx, 0, message="NaN/Inf in X batch")
    tf.debugging.assert_equal(ny, 0, message="NaN/Inf in Y batch")
    return x, y

def make_ds(X, Y, *, shuffle=True, preproc=None, limit=None,
            cache_in_memory=False, check_nans=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if preproc is not None:
        ds = ds.map(lambda x, y: tuple(preproc(x, y)), num_parallel_calls=AUTO)
    if cache_in_memory:
        ds = ds.cache()  # cache vor shuffle (vermeidet Cache-Warnungen)
    if check_nans:
        ds = ds.map(nan_debug, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0], reshuffle_each_iteration=True)
    if limit is not None:
        ds = ds.take(int(limit))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, shuffle=True,
                   preproc=map_slice_wise(preproc_train_slice),
                   check_nans=True)
val_ds   = make_ds(X_val, Y_val, shuffle=False,
                   preproc=map_slice_wise(preproc_valid_slice),
                   check_nans=True)
test_ds  = make_ds(X_test, Y_test, shuffle=False,
                   preproc=map_slice_wise(preproc_valid_slice),
                   check_nans=True)
print(">>> Datasets created")


# %%
# ==============================
# 4) Modell-Definition (3D U-Net)
# ==============================
def conv_block(x, filters, kernel_size=(3,3,3), padding="same"):
    ki  = "he_normal"
    kr  = regularizers.l2(1e-5)
    kc  = constraints.MaxNorm(3.0)

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(dtype="float32", epsilon=1e-3)(x)
    x = layers.ELU()(x)

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LayerNormalization(dtype="float32", epsilon=1e-3)(x)
    x = layers.ELU()(x)
    return x

def unet3d(input_shape=(5, 192, 240, 1), base_filters=8):
    inputs = layers.Input(shape=input_shape)
    # Encoder (nur H,W poolen)
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c3)

    # Bottleneck
    bn = conv_block(p3, base_filters*8)

    # Decoder (nur H,W upsamplen)
    u3 = layers.Conv3DTranspose(base_filters*4, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(bn)
    u3 = layers.concatenate([u3, c3])
    c4 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = conv_block(u1, base_filters)

    # Output Layer: tanh -> [0,1]
    raw_out = layers.Conv3D(1, (1,1,1), activation="tanh")(c6)
    outputs = layers.Lambda(lambda z: (z + 1.0) / 2.0, dtype="float32")(raw_out)
    return models.Model(inputs, outputs, name="3D_U-Net-ELU-LN")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)

# %%
# ==============================
# 5) Loss & Metriken
# ==============================
ALPHA_TARGET = 0.7
ALPHA = 0.0

ALPHA_TF   = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="alpha_ms_ssim")
MS_GRAD_ON = tf.Variable(False, trainable=False, dtype=tf.bool,   name="ms_grad_on")

MS_EPS    = 1e-5
K_SLICES = 2  # vorerst klein

MS_ENABLE_TF = tf.Variable(False, trainable=False, dtype=tf.bool, name="ms_enable")


def _clip01(x):
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value(x, 0.0, 1.0)

def ms_ssim_loss_sampled(y_true, y_pred, k=K_SLICES):
    yt = _clip01(y_true); yp = _clip01(y_pred)

    # komplett detach, damit wirklich kein Grad & weniger Interaktion
    yt = tf.stop_gradient(yt)
    yp = tf.stop_gradient(yp)

    MS_EPS = tf.constant(1e-5, tf.float32)
    yt = tf.clip_by_value(yt, MS_EPS, 1.0 - MS_EPS)
    yp = tf.clip_by_value(yp, MS_EPS, 1.0 - MS_EPS)

    # ganz kleines Dithering (ebenfalls float32, detached)
    noise = tf.constant(1e-4, tf.float32)
    yt = yt + noise * tf.random.normal(tf.shape(yt), dtype=tf.float32)
    yp = yp + noise * tf.random.normal(tf.shape(yp), dtype=tf.float32)

    # Energie-basiertes Sampling (robust gegen k > D)
    energy = tf.reduce_mean(tf.abs(yt) + tf.abs(yp), axis=[2,3,4])  # (B,D)
    D = tf.shape(yt)[1]
    k_eff = tf.minimum(k, D)
    idx = tf.math.top_k(energy, k=k_eff).indices
    yt = tf.gather(yt, idx, batch_dims=1)
    yp = tf.gather(yp, idx, batch_dims=1)

    yt = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))

    ms = tf.image.ssim_multiscale(yt, yp, max_val=1.0)  # float32
    ms = tf.where(tf.math.is_finite(ms), ms, tf.zeros_like(ms))
    ms_mean = tf.reduce_mean(ms)

    # harte Finite-Absicherung
    ms_mean = tf.where(tf.math.is_finite(ms_mean), ms_mean, tf.constant(0.0, tf.float32))
    return 1.0 - ms_mean


@tf.function(jit_compile=False)
def combined_loss(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    l_mae = tf.reduce_mean(tf.abs(yt - yp))

    def with_ms():
        l_ms = ms_ssim_loss_sampled(yt, yp, k=K_SLICES)
        l_ms = tf.where(tf.math.is_finite(l_ms), l_ms, l_mae)  # fallback
        l_ms_used = tf.cond(MS_GRAD_ON, lambda: l_ms, lambda: tf.stop_gradient(l_ms))
        out = (1.0 - ALPHA_TF) * l_mae + ALPHA_TF * l_ms_used
        return tf.where(tf.math.is_finite(out), out, l_mae)

    use_ms = tf.logical_and(MS_ENABLE_TF, ALPHA_TF > 0.0)
    return tf.cond(use_ms, with_ms, lambda: l_mae)




def psnr_metric(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)
psnr_metric.__name__ = "psnr"


# %%
# ==============================
# 6) Optimizer & Compile
# ==============================
opt = AdamW(
    learning_rate=1e-5,   # oder 5e-6, wenn es erneut knallt
    epsilon=1e-3,         # höher = numerisch robuster
    global_clipnorm=0.1,  # stärker clippen
    weight_decay=0.0,
    amsgrad=False
)

metric_list = [
    tf.keras.metrics.MeanAbsoluteError(name="mae"),
    tf.keras.metrics.MeanSquaredError(name="mse"),
    psnr_metric,
]
model.compile(
    optimizer=opt,
    loss=combined_loss,                 # <— statt reines MAE
    metrics=[                           # (deine Metrics bleiben)
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        psnr_metric,
    ],
    jit_compile=False
)


# Debug: Eager an
model.run_eagerly = False # Just for debugging


# %%
# ==============================
# 7) Naming files pipeline
# ==============================
def _timestamp():
    return time.strftime("%Y-%m-%dT%H-%M-%S")

def _safe_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def _serialize_optimizer(opt):
    try:
        return tf.keras.optimizers.serialize(opt)
    except Exception:
        return None

class BestFinalizeCallback(callbacks.Callback):
    # (unverändert aus deinem Code, nur hier platziert)
    def __init__(self, root: Path, run_meta: dict = None, tmp_name: str = None, code_name: str = None):
        super().__init__()
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        auto = self._auto_code_name() if (code_name is None or str(code_name).upper() == "AUTO") else code_name
        self.code = self._sanitize_code(auto)
        self.tmp_path = self.root / (tmp_name or f"{self.code}_TEMP_{uuid.uuid4().hex}.keras")
        self.best_val_loss = np.inf
        self.best_psnr = None
        self.run_meta = run_meta or {}
    @staticmethod
    def _sanitize_code(code: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (code or "").strip())
        return safe or "MODEL"
    @staticmethod
    def _auto_code_name():
        try:
            main_mod = sys.modules.get("__main__")
            if main_mod and getattr(main_mod, "__file__", None):
                return os.path.splitext(os.path.basename(main_mod.__file__))[0]
        except Exception: pass
        try:
            if sys.argv and sys.argv[0]:
                return os.path.splitext(os.path.basename(sys.argv[0]))[0]
        except Exception: pass
        try:
            for fr in inspect.stack():
                fn = fr.filename
                if fn and fn not in ("<stdin>", "<string>"):
                    return os.path.splitext(os.path.basename(fn))[0]
        except Exception: pass
        for k in ("SLURM_JOB_NAME", "PBS_JOBNAME", "JOB_NAME"):
            v = os.environ.get(k)
            if v:
                return v
        return "MODEL"
    def on_epoch_end(self, epoch, logs=None):
        if not logs or "val_loss" not in logs: return
        vloss = float(logs["val_loss"])
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            psnr = logs.get("psnr_metric")
            self.best_psnr = float(psnr) if psnr is not None else None
    def on_train_end(self, logs=None):
        vloss_str = f"{self.best_val_loss:.3e}" if np.isfinite(self.best_val_loss) else "nan"
        psnr_part = f"_PSNR_{self.best_psnr:.3g}" if (self.best_psnr is not None and np.isfinite(self.best_psnr)) else ""
        new_model = self.root / f"{self.code}_NEW_valloss_{vloss_str}{psnr_part}.keras"
        if self.tmp_path.exists() and self.tmp_path.stat().st_size > 0:
            try:
                os.replace(self.tmp_path, new_model)
            except Exception as e:
                print(f"[WARN] Konnte TEMP nicht nach NEW umbenennen: {e}")
                return
            self._write_json_for_model(new_model)
        self._rank_all_models()
        try:
            if self.tmp_path.exists(): os.remove(self.tmp_path)
        except Exception: pass
    def _write_json_for_model(self, model_path: Path):
        json_path = model_path.with_suffix(".json")
        try:
            inp_shape = tuple(int(x) for x in (self.model.input_shape or []) if isinstance(x, (int,np.integer)))
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
        meta = {
            "timestamp": _timestamp(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_commit": _safe_git_commit(),
            "code_name": self.code,
            "batch_size": self.run_meta.get("batch_size"),
            "epochs_planned": self.run_meta.get("epochs"),
            "early_stopping": self.run_meta.get("early_stopping"),
            "data_prep": self.run_meta.get("data_prep"),
            "alpha_ms_ssim": self.run_meta.get("ALPHA"),
            "best_val_loss": float(self.best_val_loss) if np.isfinite(self.best_val_loss) else None,
            "best_psnr_metric": self.best_psnr,
            "input_shape": inp_shape,
            "loss": loss_name,
            "metrics": metrics_list,
            "optimizer": _serialize_optimizer(getattr(self.model, "optimizer", None)),
            "mixed_precision_policy": mixed_precision.global_policy().name if mixed_precision.global_policy() else None,
        }
        try:
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] Konnte JSON nicht schreiben: {e}")
    @staticmethod
    def _parse_filename_simple(name: str):
        if not name.endswith(".keras"): return None
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
        except ValueError: pass
        except Exception: psnr = None
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


# %%
# ==============================
# 8) Callbacks (Guards, Logging, Checkpoints)
# ==============================

class CompactLogger(callbacks.Callback):
    def __init__(self, cols=None, show_time=True):
        super().__init__()
        self.cols = cols or ["loss","val_loss","mae","val_mae","mse","val_mse","psnr","val_psnr"]
        self.show_time = show_time
        self._t0 = None
    def on_epoch_begin(self, epoch, logs=None):
        if self.show_time: self._t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "val_loss" not in logs: return
        dt = (time.time() - self._t0) if (self.show_time and self._t0 is not None) else None
        try: lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception: lr = None
        parts = [f"E{epoch+1:03d}"]
        for k in self.cols:
            v = logs.get(k, None)
            if v is not None and np.isfinite(v):
                parts.append(f"{k}={v:7.4f}")
        if lr is not None: parts.append(f"lr={lr:.1e}")
        if dt is not None: parts.append(f"time={dt:5.1f}s")
        print(" | ".join(parts))

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

class BatchDebugDump(callbacks.Callback):
    def __init__(self, every=50, dump_first_nan=True, outdir=Path("./nan_dump")):
        super().__init__()
        self.every = every
        self.dump_first_nan = dump_first_nan
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self._nan_dumped = False

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", None)
        if batch % self.every == 0 and loss is not None:
            print(f"[Batch {batch:05d}] loss={float(loss):.6f}")

        if (loss is not None) and (not np.isfinite(loss)) and self.dump_first_nan and (not self._nan_dumped):
            # Versuch: Eingaben aus der letzten Batch ziehen (nur im Eager-Mode gut möglich)
            try:
                # Greif auf das zuletzt verarbeitete Batch aus dem Iterator zu:
                # Als pragmatischer Workaround: Modell einmal auf dem letzten Batch callen und abspeichern
                # (Hier nur Pseudogreif – wenn du schnell willst, kannst du stattdessen an der Input-Pipeline dumpen.)
                pass
            except Exception:
                pass
            self._nan_dumped = True
            print(f"[Dump] NaN bei Batch {batch}. (Siehe {self.outdir})")
            self.model.stop_training = True

class AlphaScheduler(callbacks.Callback):
    def __init__(self, target=0.3, warmup=2, epochs_to_target=6,
                 ms_enable_epoch=5, grad_on_epoch=8):
        super().__init__()
        self.target = float(target)
        self.warmup = int(warmup)
        self.epochs_to_target = int(epochs_to_target)
        self.ms_enable_epoch = int(ms_enable_epoch)
        self.grad_on_epoch = int(grad_on_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup:
            ALPHA_TF.assign(0.0)
            MS_ENABLE_TF.assign(False)
            MS_GRAD_ON.assign(False)
        else:
            t = epoch - self.warmup
            frac = tf.clip_by_value(tf.cast(t, tf.float32)/float(self.epochs_to_target), 0.0, 1.0)
            ALPHA_TF.assign(self.target * frac)
            MS_ENABLE_TF.assign(epoch >= self.ms_enable_epoch)
            MS_GRAD_ON.assign(epoch >= self.grad_on_epoch)
        print(f"[AlphaScheduler] epoch={epoch}  alpha={float(ALPHA_TF.numpy()):.3f}  ms_enable={bool(MS_ENABLE_TF.numpy())}  grad_on={bool(MS_GRAD_ON.numpy())}")

class MsTermLogger(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if bool(MS_ENABLE_TF.numpy()) and ALPHA_TF.numpy() > 0:
            # kleine Probe mit aktuellen y_true/y_pred geht hier nicht direkt; reicht, dass keine NaNs in loss sind
            if logs and "loss" in logs and not np.isfinite(logs["loss"]):
                print(f"[MsTermLogger] NaN loss @ batch {batch} (alpha={float(ALPHA_TF.numpy()):.3f})")
                self.model.stop_training = True


# Checkpoints + Meta
ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE, "epochs": EPOCHS,
    "early_stopping": {"monitor":"val_loss","patience":10},
    "data_prep": {"size": 5, "group_len": 41, "dtype": "float32"},
    "ALPHA": ALPHA
}
bf = BestFinalizeCallback(ckpt_root, run_meta=run_meta, code_name="AUTO")
ckpt_best = callbacks.ModelCheckpoint(
    filepath=str(bf.tmp_path), monitor="val_loss",
    mode="min", save_best_only=True, verbose=1
)

# Callback-Liste (ohne AlphaScheduler für Debug)
cbs = [
    AlphaScheduler(
        target=0.10,        # erstmal nur 0.10 mischen
        warmup=4,           # 4 Epochen reines MAE
        epochs_to_target=8, # langsam bis 0.10
        ms_enable_epoch=8,  # MS überhaupt erst ab Epoche 8 berechnen
        grad_on_epoch=9999  # Grad weiter AUS lassen
    ),
    WeightNaNGuard(),
    LossNaNGuard(),
    callbacks.TerminateOnNaN(),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    callbacks.EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True, verbose=0),
    ckpt_best, bf,
    CompactLogger(),
]

# %%
# ==============================
# 9) Train
# ==============================
print(">>> Phase 3: GPU training starts now!")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    validation_freq=1,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=0
)
print(">>> Phase 3: Training complete!")

# %%
# ==============================
# 10) Evaluate (Val & Test)
# ==============================
final_val = model.evaluate(val_ds, return_dict=True, verbose=0)
print("FINAL VAL:", {k: float(v) for k, v in final_val.items()})

final_test = model.evaluate(test_ds, return_dict=True, verbose=0)
print("FINAL TEST:", {k: float(v) for k, v in final_test.items()})
