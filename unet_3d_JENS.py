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

import json, socket, getpass, platform, subprocess, time, uuid # for naming files and callbacks

seed = 0
reset_random_seeds(seed)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
AUTO = tf.data.AUTOTUNE

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
EPOCHS     = 400

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




# ========= Defining 3D-U-Net Architecture ========
"""
def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x
"""

def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer="he_normal", use_bias=True)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer="he_normal", use_bias=True)(x)
    x = layers.Activation(activation)(x)
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

ALPHA = 0.7  # Weight for MS-SSIM

def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    """
    Generates deterministic matrix and samples indices using highest values per row
    """
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0]) # (B,D) matrix with random values
    topk = tf.math.top_k(rnd, k=k).indices                                 # Search for 2 highest values per row
    return topk

def _safe_imgs(y_true, y_pred):
    # mixed precision + Normalisierung robust machen
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Begrenze strikt auf [0,1] (MS-SSIM erwartet das)
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    # harte Numerik-Checks (werfen sofort Exception mit Stacktrace)
    tf.debugging.assert_all_finite(y_true, "y_true contains NaN/Inf")
    tf.debugging.assert_all_finite(y_pred, "y_pred contains NaN/Inf")
    return y_true, y_pred


def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    batch_size = tf.shape(y_true)[0]
    depth = tf.shape(y_true)[1]
    idx = _sample_depth_indices(batch_size, depth, k=k)  # (B,k)

    y_groundtruth = tf.gather(y_true, idx, batch_dims=1)  # (B,k,H,W,C)
    y_model       = tf.gather(y_pred, idx, batch_dims=1)  # (B,k,H,W,C)

    yt2 = tf.reshape(y_groundtruth, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_model,       (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))

    yt2, yp2 = _safe_imgs(yt2, yp2)
    ms = tf.image.ssim_multiscale(yt2, yp2, max_val=1.0)
    tf.debugging.assert_all_finite(ms, "MS-SSIM produced NaN/Inf")
    return 1.0 - tf.reduce_mean(ms)

"""
def combined_loss(y_true, y_pred, k_slices=1):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss_sampled(y_true, y_pred, k=k_slices)
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms
"""
    
def combined_loss(y_true, y_pred, k_slices=1):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def ms_ssim_metric(y_true, y_pred):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    yt2 = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    yt2, yp2 = _safe_imgs(yt2, yp2)
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))

def psnr_metric(y_true, y_pred):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# %%
# ======== Naming files and making callbacks =======

def _safe_git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
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
    Speichert waehrend des Runs nur in eine TEMP-Datei (save_best_only=True).
    Am Ende: legt neue Datei mit Metriken an (NEW_...), ranked alle Modelle zu V1,V2,...
    und schreibt daneben eine JSON: <Modellname_ohne_Ext>_<Datum>.json
    """
    def __init__(self, root: Path, run_meta: dict = None, tmp_name: str = None):
        super().__init__()
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.root / (tmp_name or f"TEMP_{uuid.uuid4().hex}.keras")
        self.best_val_loss = np.inf
        self.best_psnr = None
        self.run_meta = run_meta or {}

    def on_epoch_end(self, epoch, logs=None):
        if not logs or "val_loss" not in logs: return
        vloss = float(logs["val_loss"])
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            psnr = logs.get("psnr_metric")
            self.best_psnr = float(psnr) if psnr is not None else None

    def on_train_end(self, logs=None):
        # 1) TEMP -> NEW_* mit Metriken
        vloss_str = f"{self.best_val_loss:.3e}" if np.isfinite(self.best_val_loss) else "nan"
        psnr_part = f"_PSNR_{self.best_psnr:.3g}" if (self.best_psnr is not None and np.isfinite(self.best_psnr)) else ""
        new_model = self.root / f"NEW_valloss_{vloss_str}{psnr_part}.keras"
        if self.tmp_path.exists():
            os.replace(self.tmp_path, new_model)

        # 2) JSON fuer dieses neu entstandene Modell schreiben (mit Datum)
        self._write_json_for_model(new_model)

        # 3) Globales Ranking aller Modelle im Ordner: V1, V2, ...
        self._rank_all_models()

    # ---------- JSON ----------
    def _write_json_for_model(self, model_path: Path):
        json_path = model_path.with_suffix(".json")

        meta = {
            "timestamp": _timestamp(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_commit": _safe_git_commit(),

            # Trainings-Hparam aus run_meta
            "batch_size": self.run_meta.get("batch_size"),
            "epochs_planned": self.run_meta.get("epochs"),
            "early_stopping": self.run_meta.get("early_stopping"),
            "data_prep": self.run_meta.get("data_prep"),
            "alpha_ms_ssim": self.run_meta.get("ALPHA"),

            # Beste Metriken dieses Runs
            "best_val_loss": float(self.best_val_loss) if np.isfinite(self.best_val_loss) else None,
            "best_psnr_metric": self.best_psnr,

            # Modell/Compile-Zustand
            "input_shape": tuple(int(x) for x in (self.model.input_shape or []) if isinstance(x, (int,np.integer))),
            "loss": getattr(self.model.loss, "__name__", str(self.model.loss)),
            "metrics": [getattr(m, "__name__", str(m)) for m in (self.model.metrics or [])],
            "optimizer": _serialize_optimizer(self.model.optimizer),
            "mixed_precision_policy": mixed_precision.global_policy().name if mixed_precision.global_policy() else None,
        }
        try:
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] Konnte JSON nicht schreiben: {e}")

    # ---------- Ranking & Umbenennen ----------
    @staticmethod
    def _parse_filename_simple(name: str):
        if not name.endswith(".keras"): return None
        base = name[:-6]; parts = base.split("_")
        if "valloss" not in parts: return None
        val_loss = None; psnr = None
        for i,p in enumerate(parts):
            if p=="valloss" and i+1<len(parts):
                try: val_loss = float(parts[i+1])
                except: return None
            if p=="PSNR" and i+1<len(parts):
                try: psnr = float(parts[i+1])
                except: psnr = None
        if val_loss is None: return None
        return {"val_loss": val_loss, "psnr": psnr}

    def _rank_all_models(self):
        items = []
        for p in self.root.iterdir():
            if p.is_file() and p.suffix == ".keras":
                m = self._parse_filename_simple(p.name)
                if m:
                    items.append((p, m["val_loss"], m["psnr"]))
        if not items:
            return

        items.sort(key=lambda x: (x[1], x[0].stat().st_mtime))

        temps = []
        for path, vloss, psnr in items:
            # alle JSONs finden, die zu diesem Modell gehoeren (mit Timestamp)
            base_stem = path.with_suffix("").name  # ohne .keras
            jsons = list(self.root.glob(base_stem + "_*.json"))

            t_model = self.root / f".tmp_{uuid.uuid4().hex}.keras"
            os.replace(path, t_model)

            # zugehoerige JSONs -> passende tmp-Namen mit gleicher Timestamp
            tmp_jsons = []
            for j in jsons:
                ts_suffix = j.name[len(base_stem):]  # z.B. "_2025-09-22T12-05-31.json"
                t_json = t_model.with_suffix("")  # .tmp_<id>
                t_json = t_json.parent / (t_json.name + ts_suffix)  # .tmp_<id>_TIMESTAMP.json
                os.replace(j, t_json)
                tmp_jsons.append((t_json, ts_suffix))

            temps.append((t_model, tmp_jsons, vloss, psnr))

        for rank, (t_model, tmp_jsons, vloss, psnr) in enumerate(temps, start=1):
            v = f"{vloss:.3e}"
            ps = f"_PSNR_{psnr:.3g}" if psnr is not None else ""
            final_model = self.root / f"V{rank}_valloss_{v}{ps}.keras"
            os.replace(t_model, final_model)

            # JSONs zum finalen Basenamen + unveraendertem Timestamp umbenennen
            final_stem = final_model.with_suffix("").name
            for t_json, ts_suffix in tmp_jsons:
                if t_json.exists():
                    final_json = final_model.with_suffix("")  # ohne .keras
                    final_json = final_json.parent / (final_stem + ts_suffix)  # V{rank}_..._TIMESTAMP.json
                    os.replace(t_json, final_json)



# %%
# ======== Train =======

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

import json, socket, getpass, platform, subprocess, time, uuid # for naming files and callbacks

seed = 0
reset_random_seeds(seed)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
AUTO = tf.data.AUTOTUNE

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
EPOCHS     = 400

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




# ========= Defining 3D-U-Net Architecture ========
"""
def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x
"""

def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer="he_normal", use_bias=True)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding,
                      kernel_initializer="he_normal", use_bias=True)(x)
    x = layers.Activation(activation)(x)
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


# =========== Defining Loss function MAE + MS-SSIM (slice-wise) ========
def _sample_depth_indices(batch_size, depth, k=1, seed=42):
    """
    Generates deterministic matrix and samples indices using highest values per row
    """
    rnd = tf.random.stateless_uniform([batch_size, depth], seed=[seed, 0]) # (B,D) matrix with random values
    topk = tf.math.top_k(rnd, k=k).indices                                 # Search for 2 highest values per row
    return topk

def _safe_imgs(y_true, y_pred):
    # mixed precision + Normalisierung robust machen
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Begrenze strikt auf [0,1] (MS-SSIM erwartet das)
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    # harte Numerik-Checks (werfen sofort Exception mit Stacktrace)
    tf.debugging.assert_all_finite(y_true, "y_true contains NaN/Inf")
    tf.debugging.assert_all_finite(y_pred, "y_pred contains NaN/Inf")
    return y_true, y_pred


def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    batch_size = tf.shape(y_true)[0]
    depth = tf.shape(y_true)[1]
    idx = _sample_depth_indices(batch_size, depth, k=k)  # (B,k)

    y_groundtruth = tf.gather(y_true, idx, batch_dims=1)  # (B,k,H,W,C)
    y_model       = tf.gather(y_pred, idx, batch_dims=1)  # (B,k,H,W,C)

    yt2 = tf.reshape(y_groundtruth, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_model,       (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))

    yt2, yp2 = _safe_imgs(yt2, yp2)
    ms = tf.image.ssim_multiscale(yt2, yp2, max_val=1.0)
    tf.debugging.assert_all_finite(ms, "MS-SSIM produced NaN/Inf")
    return 1.0 - tf.reduce_mean(ms)

ALPHA = 0.3  # erstmal kleiner starten

def combined_loss(y_true, y_pred, k_slices=1):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss_sampled(y_true, y_pred, k=k_slices)  # nutzt _safe_imgs intern
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms

def ms_ssim_metric(y_true, y_pred):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    yt2 = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    yt2, yp2 = _safe_imgs(yt2, yp2)
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))

def psnr_metric(y_true, y_pred):
    y_true, y_pred = _safe_imgs(y_true, y_pred)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)
# ======== Naming files and making callbacks =======

def _safe_git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
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
    Speichert waehrend des Runs nur in eine TEMP-Datei (save_best_only=True).
    Am Ende: legt neue Datei mit Metriken an (NEW_...), ranked alle Modelle zu V1,V2,...
    und schreibt daneben eine JSON: <Modellname_ohne_Ext>_<Datum>.json
    """
    def __init__(self, root: Path, run_meta: dict = None, tmp_name: str = None):
        super().__init__()
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.root / (tmp_name or f"TEMP_{uuid.uuid4().hex}.keras")
        self.best_val_loss = np.inf
        self.best_psnr = None
        self.run_meta = run_meta or {}

    def on_epoch_end(self, epoch, logs=None):
        if not logs or "val_loss" not in logs: return
        vloss = float(logs["val_loss"])
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            psnr = logs.get("psnr_metric")
            self.best_psnr = float(psnr) if psnr is not None else None

    def on_train_end(self, logs=None):
        # 1) TEMP -> NEW_* mit Metriken
        vloss_str = f"{self.best_val_loss:.3e}" if np.isfinite(self.best_val_loss) else "nan"
        psnr_part = f"_PSNR_{self.best_psnr:.3g}" if (self.best_psnr is not None and np.isfinite(self.best_psnr)) else ""
        new_model = self.root / f"NEW_valloss_{vloss_str}{psnr_part}.keras"
        if self.tmp_path.exists():
            os.replace(self.tmp_path, new_model)

        # 2) JSON fuer dieses neu entstandene Modell schreiben (mit Datum)
        self._write_json_for_model(new_model)

        # 3) Globales Ranking aller Modelle im Ordner: V1, V2, ...
        self._rank_all_models()

    # ---------- JSON ----------
    def _write_json_for_model(self, model_path: Path):
        json_path = model_path.with_suffix(".json")

        meta = {
            "timestamp": _timestamp(),
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "git_commit": _safe_git_commit(),

            # Trainings-Hparam aus run_meta
            "batch_size": self.run_meta.get("batch_size"),
            "epochs_planned": self.run_meta.get("epochs"),
            "early_stopping": self.run_meta.get("early_stopping"),
            "data_prep": self.run_meta.get("data_prep"),
            "alpha_ms_ssim": self.run_meta.get("ALPHA"),

            # Beste Metriken dieses Runs
            "best_val_loss": float(self.best_val_loss) if np.isfinite(self.best_val_loss) else None,
            "best_psnr_metric": self.best_psnr,

            # Modell/Compile-Zustand
            "input_shape": tuple(int(x) for x in (self.model.input_shape or []) if isinstance(x, (int,np.integer))),
            "loss": getattr(self.model.loss, "__name__", str(self.model.loss)),
            "metrics": [getattr(m, "__name__", str(m)) for m in (self.model.metrics or [])],
            "optimizer": _serialize_optimizer(self.model.optimizer),
            "mixed_precision_policy": mixed_precision.global_policy().name if mixed_precision.global_policy() else None,
        }
        try:
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[WARN] Konnte JSON nicht schreiben: {e}")

    # ---------- Ranking & Umbenennen ----------
    @staticmethod
    def _parse_filename_simple(name: str):
        if not name.endswith(".keras"): return None
        base = name[:-6]; parts = base.split("_")
        if "valloss" not in parts: return None
        val_loss = None; psnr = None
        for i,p in enumerate(parts):
            if p=="valloss" and i+1<len(parts):
                try: val_loss = float(parts[i+1])
                except: return None
            if p=="PSNR" and i+1<len(parts):
                try: psnr = float(parts[i+1])
                except: psnr = None
        if val_loss is None: return None
        return {"val_loss": val_loss, "psnr": psnr}

    def _rank_all_models(self):
        items = []
        for p in self.root.iterdir():
            if p.is_file() and p.suffix == ".keras":
                m = self._parse_filename_simple(p.name)
                if m:
                    items.append((p, m["val_loss"], m["psnr"]))
        if not items:
            return

        items.sort(key=lambda x: (x[1], x[0].stat().st_mtime))

        temps = []
        for path, vloss, psnr in items:
            # alle JSONs finden, die zu diesem Modell gehoeren (mit Timestamp)
            base_stem = path.with_suffix("").name  # ohne .keras
            jsons = list(self.root.glob(base_stem + "_*.json"))

            t_model = self.root / f".tmp_{uuid.uuid4().hex}.keras"
            os.replace(path, t_model)

            # zugehoerige JSONs -> passende tmp-Namen mit gleicher Timestamp
            tmp_jsons = []
            for j in jsons:
                ts_suffix = j.name[len(base_stem):]  # z.B. "_2025-09-22T12-05-31.json"
                t_json = t_model.with_suffix("")  # .tmp_<id>
                t_json = t_json.parent / (t_json.name + ts_suffix)  # .tmp_<id>_TIMESTAMP.json
                os.replace(j, t_json)
                tmp_jsons.append((t_json, ts_suffix))

            temps.append((t_model, tmp_jsons, vloss, psnr))

        for rank, (t_model, tmp_jsons, vloss, psnr) in enumerate(temps, start=1):
            v = f"{vloss:.3e}"
            ps = f"_PSNR_{psnr:.3g}" if psnr is not None else ""
            final_model = self.root / f"V{rank}_valloss_{v}{ps}.keras"
            os.replace(t_model, final_model)

            # JSONs zum finalen Basenamen + unveraendertem Timestamp umbenennen
            final_stem = final_model.with_suffix("").name
            for t_json, ts_suffix in tmp_jsons:
                if t_json.exists():
                    final_json = final_model.with_suffix("")  # ohne .keras
                    final_json = final_json.parent / (final_stem + ts_suffix)  # V{rank}_..._TIMESTAMP.json
                    os.replace(t_json, final_json)



# %%
# ======== Train (STEP 1: MAE-only sanity) ========
print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)

# MAE-only Loss für Stabilitätscheck
def mae_only(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, epsilon=1e-7)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0, epsilon=1e-7),
    loss=combined_loss,
    metrics=["mae", psnr_metric, ms_ssim_metric, ms_ssim_loss_metric],  # <- optional
    jit_compile=False
)

# Forward-Check (ohne Training): produziert das Netz NaNs?
xb, yb = next(iter(train_ds.take(1)))
y_eval  = model(xb, training=False)
tf.debugging.assert_all_finite(y_eval, "FORWARD(training=False) produced NaN/Inf")
y_train = model(xb, training=True)
tf.debugging.assert_all_finite(y_train, "FORWARD(training=True) produced NaN/Inf")
print("forward ok:",
      float(tf.reduce_min(y_eval)), float(tf.reduce_max(y_eval)),
      float(tf.reduce_min(y_train)), float(tf.reduce_max(y_train)))

# Mini-Fit: sehr kurz, nur zum Sanity-Check
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=[tf.keras.callbacks.TerminateOnNaN()],
    verbose=2
)



print(">>> Phase 3: Training complete!")
