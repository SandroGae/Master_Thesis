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

from tensorflow.keras import regularizers, constraints, layers, models, callbacks
from tensorflow.keras.optimizers import AdamW

from unet_3d_data_JENS import prepare_in_memory_5to5
from jens_stuff import SumScaleNormalizer, reset_random_seeds

# Reproduzierbarkeit
seed = 0
reset_random_seeds(seed)

# Lädt Daten in GPU dynamisch nach Bedarf
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

AUTO = tf.data.AUTOTUNE

# %%
# ==============================
# 1) Daten laden (CPU)
# ==============================

print(">>> Phase 1: Starting data prep on CPU...")

# Daten werden ins RAM geladen
results = prepare_in_memory_5to5(data_dir=Path.home() / "data" / "original_data",
    size=5, group_len=41, dtype=np.float32,)

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

# Normalisierung auf Summe 1.0 pro Slice (H,W,C), getrennt für Train/Val
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
# 4) Modell-Definition (3D U-Net) – FLACH & STABIL
# ==============================
def conv_block(x, filters, k=(1,3,3), p="same", drop=0.05):
    """Conv3D -> BN -> LeakyReLU (x2) + Residual, optional SpatialDropout3D."""
    kr = regularizers.l2(1e-6)
    kc = constraints.MaxNorm(2.0)

    def conv_bn_lrelu(z, f):
        z = layers.Conv3D(
            f, k, padding=p, use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kr, kernel_constraint=kc
        )(z)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU(negative_slope=0.1)
        return z

    y = conv_bn_lrelu(x, filters)
    y = conv_bn_lrelu(y, filters)

    if drop and drop > 0.0:
        y = layers.SpatialDropout3D(drop)(y)

    # Residual Pfad: falls Kanäle nicht passen, 1x1x1-Projektion
    if x.shape[-1] != filters:
        x = layers.Conv3D(filters, (1,1,1), padding=p, use_bias=False)(x)
        x = layers.BatchNormalization()(x)

    return layers.Add()([x, y])


def unet3d(input_shape=(5, 192, 240, 1), base_filters=8):
    """Sehr flaches 3D-UNet: 2 Down-/Up-Stufen, nur H/W-Operationen."""
    inputs = layers.Input(shape=input_shape)

    # Encoder (nur H,W poolen)
    c1 = conv_block(inputs, base_filters,   k=(1,3,3), drop=0.05)
    p1 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c1)

    c2 = conv_block(p1,     base_filters*2, k=(1,3,3), drop=0.05)
    p2 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c2)

    # Bottleneck
    bn = conv_block(p2,     base_filters*4, k=(1,3,3), drop=0.05)

    # Decoder (nur H,W upsamplen)
    u2 = layers.Conv3DTranspose(base_filters*2, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(bn)
    u2 = layers.Concatenate()([u2, c2])
    c3 = conv_block(u2, base_filters*2, k=(1,3,3), drop=0.05)

    u1 = layers.Conv3DTranspose(base_filters,   kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = conv_block(u1, base_filters, k=(1,3,3), drop=0.05)

    # Output: Sigmoid in [0,1]
    out = layers.Conv3D(1, (1,1,1), activation="sigmoid")(c4)
    out = layers.Lambda(lambda z: tf.clip_by_value(tf.cast(z, tf.float32), 0.0, 1.0))(out)

    return models.Model(inputs, out, name="UNet3D")

# Modell instanziieren – SCHMAL & FLACH
model = unet3d(input_shape=INPUT_SHAPE, base_filters=8)


# %%
# ==============================
# 5) Loss & Metriken
# ==============================
ALPHA_TARGET = 0.0
ALPHA = 0.0

ALPHA_TF   = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="alpha_ms_ssim")
MS_GRAD_ON = tf.Variable(False, trainable=False, dtype=tf.bool,   name="ms_grad_on")

MS_ENABLE_TF = tf.Variable(False, trainable=False, dtype=tf.bool, name="ms_enable")


def _clip01(x):
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value(x, 0.0, 1.0)

def ms_ssim_loss_all_slices(y_true, y_pred):
    # Eingabe: (B, D, H, W, C)  -> wir werten alle D-Slices gleichberechtigt aus
    yt = _clip01(y_true)
    yp = _clip01(y_pred)

    MS_EPS = tf.constant(1e-5, tf.float32)
    yt = tf.clip_by_value(yt, MS_EPS, 1.0 - MS_EPS)
    yp = tf.clip_by_value(yp, MS_EPS, 1.0 - MS_EPS)

    # (optional) leichtes Dithering – kann man entfernen, falls du willst
    noise = tf.constant(1e-4, tf.float32)
    yt = yt + noise * tf.random.normal(tf.shape(yt), dtype=tf.float32)
    yp = yp + noise * tf.random.normal(tf.shape(yp), dtype=tf.float32)

    # Alle Slices als 2D-Bilder behandeln: (B*D, H, W, C)
    yt = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))

    ms = tf.image.ssim_multiscale(yt, yp, max_val=1.0)  # (B*D,)
    ms = tf.where(tf.math.is_finite(ms), ms, tf.zeros_like(ms))
    ms_mean = tf.reduce_mean(ms)  # Mittelwert über alle Slices und Batch

    return 1.0 - ms_mean


@tf.function(jit_compile=False)
def combined_loss(y_true, y_pred):
    yt = _clip01(y_true)
    yp = _clip01(y_pred)
    l_mae = tf.reduce_mean(tf.abs(yt - yp))

    def with_ms():
        l_ms = ms_ssim_loss_all_slices(yt, yp)          # <-- NEU
        l_ms = tf.where(tf.math.is_finite(l_ms), l_ms, l_mae)  # Fallback
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
opt = AdamW(learning_rate=5e-6, epsilon=1e-3,
            global_clipnorm=0.05, weight_decay=1e-4, amsgrad=True)


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
            psnr = logs.get("psnr")
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

class AlphaScheduler(callbacks.Callback):
    def __init__(self, step=0.02, target=0.7, ms_enable_epoch=0, grad_on_epoch=8):
        super().__init__()
        self.step = float(step)
        self.target = float(target)
        self.ms_enable_epoch = int(ms_enable_epoch)
        self.grad_on_epoch = int(grad_on_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        # epoch startet bei 0 -> schon in Epoche 1 (epoch=0) alpha=0.02
        a = (epoch + 1) * self.step
        a = min(a, self.target)
        ALPHA_TF.assign(a)

        # MS-SSIM ab Start erlauben, aber Gradienten z.B. erst ab Ep. 9 (anpassbar)
        MS_ENABLE_TF.assign(epoch >= self.ms_enable_epoch)
        MS_GRAD_ON.assign(epoch >= self.grad_on_epoch)

        print(f"[AlphaScheduler] epoch={epoch}  alpha={float(ALPHA_TF.numpy()):.3f}  "
              f"ms_enable={bool(MS_ENABLE_TF.numpy())}  grad_on={bool(MS_GRAD_ON.numpy())}")

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

# Callback-Liste
cbs = [
    AlphaScheduler(step=0.02, target=0.7, ms_enable_epoch=0, grad_on_epoch=8),
    WeightNaNGuard(), LossNaNGuard(), callbacks.TerminateOnNaN(),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
    callbacks.EarlyStopping(monitor="val_loss", patience=16, restore_best_weights=True, verbose=0),
    ckpt_best, bf, CompactLogger(),
]

# %%
# ==============================
# 9) Train
# ==============================
print(">>> Phase 3: GPU training starts now!")
history = model.fit(
    train_ds,
    validation_data=val_ds,
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
