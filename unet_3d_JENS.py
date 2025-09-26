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
# ======== Imports =======
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32") # float 32 to test if this was the issue
from tensorflow.keras import layers, models, callbacks
from unet_3d_data_JENS import prepare_in_memory_5to5
from jens_stuff import SumScaleNormalizer, reset_random_seeds
from pathlib import Path
from tensorflow.keras import regularizers, constraints

import sys, inspect
import json, socket, getpass, platform, subprocess, time, uuid # for naming files and callbacks

seed = 0
reset_random_seeds(seed)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
AUTO = tf.data.AUTOTUNE

# %%
# ===== Loading Data in RAM =====
print(">>> Phase 1: Starting data prep on CPU...")
results = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    size=5, group_len=41, dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")
X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]

INPUT_SHAPE = X_train.shape[1:]
BATCH_SIZE = 16
EPOCHS     = 10

# %%
# ===== Preprocessing (nach dem Laden definieren) =====
preproc_train = SumScaleNormalizer(
    scale_range=[5000, 15001], pre_offset=0.0,
    normalize_label=True, axis=None, batch_mode=False,
    clip_before=[0., np.inf], clip_after=[0., 1.]
)
preproc_valid = SumScaleNormalizer(
    scale_range=[5000, 5001], pre_offset=0.0,
    normalize_label=True, axis=None, batch_mode=False,
    clip_before=[0., np.inf], clip_after=[0., 1.]
)

def make_ds(X, Y, shuffle=True, preproc=None):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if preproc is not None:
        ds = ds.map(lambda x, y: tuple(preproc.map(x, y)),
                    num_parallel_calls=AUTO).cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, True,  preproc=preproc_train)
val_ds   = make_ds(X_val,   Y_val,   False, preproc=preproc_valid)
test_ds  = make_ds(X_test,  Y_test,  False, preproc=preproc_valid)
print(">>> Datasets created")


# %%
# ========= Defining 3D-U-Net Architecture ========

def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation=None):
    ki  = "glorot_uniform"
    kr  = regularizers.l2(1e-5)                  # milde L2
    kc  = constraints.MaxNorm(3.0)               # Max-Norm gegen Explodieren

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)           # sanfter als ReLU

    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer=ki, use_bias=True,
                      kernel_regularizer=kr, kernel_constraint=kc)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def unet3d(input_shape=(5, 192, 240, 1), base_filters=32):
    inputs = layers.Input(shape=input_shape)

    # Encoder (pool only over H,W)
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(c3)

    # Bottleneck
    bn = conv_block(p3, base_filters*8)

    # Decoder (upsample only over H,W)
    u3 = layers.Conv3DTranspose(base_filters*4, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(bn) # bottleneck
    u3 = layers.concatenate([u3, c3])
    c4 = conv_block(u3, base_filters*4)

    u2 = layers.Conv3DTranspose(base_filters*2, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, base_filters*2)

    u1 = layers.Conv3DTranspose(base_filters, kernel_size=(1,2,2), strides=(1,2,2), padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = conv_block(u1, base_filters)

    outputs = layers.Conv3D(1, (1,1,1), dtype="float32", activation="sigmoid")(c6)
    return models.Model(inputs, outputs, name="3D_U-Net")


# %%
# =========== Defining Loss function MAE + MS-SSIM (slice-wise) ========

ALPHA_TARGET = 0.7    # <- bis hier hochfahren
ALPHA = 0.0
K_SLICES     = 5
MS_EPS       = 1e-5

# ALPHA als TF-Variable (damit der Graph nicht einfriert)
ALPHA_TF = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="alpha_ms_ssim")


def _clip01(x):
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value(x, 0.0, 1.0)

def _safe_imgs_for_ms_ssim(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    yt = tf.clip_by_value(yt, MS_EPS, 1.0 - MS_EPS)
    yp = tf.clip_by_value(yp, MS_EPS, 1.0 - MS_EPS)
    tf.debugging.assert_all_finite(yt, "y_true contains NaN/Inf")
    tf.debugging.assert_all_finite(yp, "y_pred contains NaN/Inf")
    return yt, yp

def ms_ssim_loss_sampled(y_true, y_pred, k=K_SLICES):
    yt, yp = _safe_imgs_for_ms_ssim(y_true, y_pred)
    B = tf.shape(yt)[0]; D = tf.shape(yt)[1]
    idx = _sample_depth_indices(B, D, k=k)
    ygt = tf.gather(yt, idx, batch_dims=1)
    ypd = tf.gather(yp, idx, batch_dims=1)
    yt2 = tf.reshape(ygt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp2 = tf.reshape(ypd, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))
    ms = tf.image.ssim_multiscale(yt2, yp2, max_val=1.0)
    tf.debugging.assert_all_finite(ms, "MS-SSIM produced NaN/Inf")
    return 1.0 - tf.reduce_mean(ms)

def combined_loss(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    l_mae = tf.reduce_mean(tf.abs(yt - yp))
    def ms_branch():
        l_ms = ms_ssim_loss_sampled(yt, yp, k=K_SLICES)
        return (1.0 - ALPHA_TF) * l_mae + ALPHA_TF * l_ms
    def mae_branch():
        return l_mae
    return tf.cond(ALPHA_TF > 0.0, ms_branch, mae_branch)

def mae_metric(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    return tf.reduce_mean(tf.abs(yt - yp))

def ms_ssim_metric(y_true, y_pred):
    yt, yp = _safe_imgs_for_ms_ssim(y_true, y_pred)
    yt2 = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp2 = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))
def psnr_metric(y_true, y_pred):
    yt = _clip01(y_true); yp = _clip01(y_pred)
    return tf.image.psnr(yt, yp, max_val=1.0)

def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    """
    Generates deterministic matrix and samples indices using highest values per row
    """
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0]) # (B,D) matrix with random values
    topk = tf.math.top_k(rnd, k=k).indices                                 # Search for 2 highest values per row
    return topk



# %%
# ======== Automaic Naming Pipeline ============

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

def _timestamp():
    # Dateiname-sicher (kein ":" unter Windows)
    return time.strftime("%Y-%m-%dT%H-%M-%S")

class BestFinalizeCallback(callbacks.Callback):
    """
    Am Ende:
      1) nimmt die von ModelCheckpoint geschriebene TEMP-Datei und benennt sie um zu <code>_NEW_valloss_...>.keras
      2) schreibt JSON
      3) rankt *alle* <code>_*.keras strikt nach val_loss -> <code>_V1_..., <code>_V2_..., ... (Luecken werden beseitigt)
    """
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
        # 1) __main__.__file__
        try:
            main_mod = sys.modules.get("__main__")
            if main_mod and getattr(main_mod, "__file__", None):
                return os.path.splitext(os.path.basename(main_mod.__file__))[0]
        except Exception:
            pass
        # 2) sys.argv[0]
        try:
            if sys.argv and sys.argv[0]:
                return os.path.splitext(os.path.basename(sys.argv[0]))[0]
        except Exception:
            pass
        # 3) erster echter Stack-Frame
        try:
            for fr in inspect.stack():
                fn = fr.filename
                if fn and fn not in ("<stdin>", "<string>"):
                    return os.path.splitext(os.path.basename(fn))[0]
        except Exception:
            pass
        # 4) SLURM/PBS
        for k in ("SLURM_JOB_NAME", "PBS_JOBNAME", "JOB_NAME"):
            v = os.environ.get(k)
            if v:
                return v
        return "MODEL"

    # ---------- Trainingslogik: nur Metriken merken, KEIN Speichern hier! ----------
    def on_epoch_end(self, epoch, logs=None):
        if not logs or "val_loss" not in logs:
            return
        vloss = float(logs["val_loss"])
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            psnr = logs.get("psnr_metric")
            self.best_psnr = float(psnr) if psnr is not None else None

    def on_train_end(self, logs=None):
        # TEMP -> NEW_* (nur wenn TEMP existiert – ModelCheckpoint muss sie geschrieben haben)
        vloss_str = f"{self.best_val_loss:.3e}" if np.isfinite(self.best_val_loss) else "nan"
        psnr_part = f"_PSNR_{self.best_psnr:.3g}" if (self.best_psnr is not None and np.isfinite(self.best_psnr)) else ""
        new_model = self.root / f"{self.code}_NEW_valloss_{vloss_str}{psnr_part}.keras"

        if self.tmp_path.exists() and self.tmp_path.stat().st_size > 0:
            try:
                os.replace(self.tmp_path, new_model)
            except Exception as e:
                print(f"[WARN] Konnte TEMP nicht nach NEW umbenennen: {e}")
                return
            # JSON schreiben
            self._write_json_for_model(new_model)

        # Re-Ranking fuer alle mit diesem Prefix
        self._rank_all_models()

        # Aufraeumen
        try:
            if self.tmp_path.exists():
                os.remove(self.tmp_path)
        except Exception:
            pass

    # ---------- JSON ----------
    def _write_json_for_model(self, model_path: Path):
        # kein Timestamp mehr im Dateinamen
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
            "timestamp": _timestamp(),   # bleibt im JSON-Inhalt erhalten!
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

    # ---------- Parsing ----------
    @staticmethod
    def _parse_filename_simple(name: str):
        if not name.endswith(".keras"):
            return None
        base = name[:-6]
        parts = base.split("_")
        try:
            i_vl = parts.index("valloss")
        except ValueError:
            return None
        if i_vl + 1 >= len(parts):
            return None
        try:
            val_loss = float(parts[i_vl + 1])
        except Exception:
            return None
        psnr = None
        try:
            i_ps = parts.index("PSNR")
            if i_ps + 1 < len(parts):
                psnr = float(parts[i_ps + 1])
        except ValueError:
            pass
        except Exception:
            psnr = None
        return {"val_loss": val_loss, "psnr": psnr}

    # ---------- Ranking & Umbenennen ----------
    def _rank_all_models(self):
        items = []
        for p in self.root.glob(f"{self.code}_*.keras"):
            if not p.is_file():
                continue
            meta = self._parse_filename_simple(p.name)
            if meta:
                items.append((p, meta["val_loss"], meta["psnr"]))
        if not items:
            return
        items.sort(key=lambda x: (x[1], x[0].stat().st_mtime))

        temps = []
        for path, vloss, psnr in items:
            base_stem = path.with_suffix("").name
            jsons = []
            p0 = self.root / (base_stem + ".json")
            if p0.exists():
                jsons.append(p0)
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
# ======== Train (STEP 1: MAE-only sanity) ========
print(">>> Phase 3: GPU training starts now!")

# ======== Alpha linear hochfahren ========
class AlphaScheduler(callbacks.Callback):
    def __init__(self, warmup=1, step=0.1, target=ALPHA_TARGET, min_alpha=0.0):
        super().__init__()
        self.warmup = int(warmup)
        self.step   = float(step)
        self.target = float(target)
        self.min_a  = float(min_alpha)
        self.best_val = np.inf

    def on_epoch_begin(self, epoch, logs=None):
        # während Warmup bei 0.0 bleiben
        if epoch < self.warmup:
            ALPHA_TF.assign(self.min_a)
        print(f"[AlphaScheduler] begin epoch {epoch}  ALPHA={float(ALPHA_TF.numpy()):.3f}")

    def on_epoch_end(self, epoch, logs=None):
        vl = float(logs.get("val_loss", np.inf))
        if not np.isfinite(vl):
            # instabil -> alpha leicht senken
            new_a = max(self.min_a, float(ALPHA_TF.numpy()) - self.step/2.0)
            ALPHA_TF.assign(new_a)
            print(f"[AlphaScheduler] val_loss non-finite -> ALPHA={new_a:.3f}")
            return

        improved = vl < self.best_val - 1e-6
        if improved:
            self.best_val = vl

        # Wenn stabil (endlich) und nicht deutlich schlechter, erhöhe ALPHA
        if vl <= self.best_val * 1.02:  # <= +2% Verschlechterung tolerieren
            new_a = min(self.target, float(ALPHA_TF.numpy()) + self.step)
            ALPHA_TF.assign(new_a)
            print(f"[AlphaScheduler] val ok -> ALPHA={new_a:.3f}")
        else:
            # leichte Verschlechterung -> halte oder kleine Reduktion
            new_a = max(self.min_a, float(ALPHA_TF.numpy()) - self.step/2.0)
            ALPHA_TF.assign(new_a)
            print(f"[AlphaScheduler] val worsened -> ALPHA={new_a:.3f}")

# ======== Checkpoints inkl. BestFinalize ========
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

# ======== Callbacks-Liste ========

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)

# ======== Optimizer (etwas konservativer) ========
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-5,       # notfalls 5e-6
    epsilon=1e-6,
    global_clipnorm=1.0,      # GANZ WICHTIG: global, nicht nur per-layer
    clipvalue=0.5             # optional: zusätzlich kappen
)
model.compile(optimizer=opt, loss=combined_loss,
              metrics=["mae", psnr_metric, ms_ssim_metric], jit_compile=False)

model.run_eagerly = True  # nur zum Debuggen, danach wieder False

class LogAlphaAtEnd(callbacks.Callback):
    def on_train_end(self, logs=None):
        try:
            bf.run_meta["ALPHA_final"] = float(ALPHA_TF.numpy())
            bf.run_meta["ALPHA_target"] = float(ALPHA_TARGET)
        except Exception:
            pass

class WeightNaNGuard(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        for i, w in enumerate(self.model.weights):
            if not tf.reduce_all(tf.math.is_finite(w)):
                print(f"[NaNWeight] layer={w.name} batch={batch}")
                self.model.stop_training = True
                break

cbs = [
    AlphaScheduler(warmup=2, step=0.05, target=ALPHA_TARGET),
    WeightNaNGuard(),
    LogAlphaAtEnd(),                 # <- HINZU
    callbacks.TerminateOnNaN(),
    ckpt_best, bf,
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
]

# ======== Train ========
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,   # z.B. 6-10 empfehlenswert fuer den Ramp
    callbacks=cbs,
    verbose=2
)


print(">>> Phase 3: Training complete!")

