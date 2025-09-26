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
# ======== Imports =======
import os

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") # Increases performance without loss of quality (calculations still done with float_32 precision)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from unet_3d_data import prepare_in_memory_5to5
from pathlib import Path

import sys, inspect
import json, socket, getpass, platform, subprocess, time, uuid # for naming files and callbacks


# %%
# ======== Allocate GPU memory dynamically as needed =======
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

AUTO = tf.data.AUTOTUNE # Chooses optimal number of threads automatically depending on hardware

# %%
# ===== Loading Data in RAM =====

print(">>> Phase 1: Starting data prep on CPU...")
(results, size) = prepare_in_memory_5to5(
    data_dir=Path.home() / "data" / "original_data",
    use_vst=False, # No anscombe transform
    size=5,
    group_len=41,
    dtype=np.float32,
)
print(">>> Data preperation finished, all data in RAM")

X_train, Y_train = results["train"]
X_val,   Y_val   = results["val"]
X_test,  Y_test  = results["test"]


INPUT_SHAPE = X_train.shape[1:]  # (5, H, W, 1)

# %%
# ======== Making Tensorflow dataset =======

BATCH_SIZE = 16
EPOCHS     = 3

# Sanity check for INPUT_SHAPE
D,H,W,C = INPUT_SHAPE
if (H % 8) or (W % 8):
    print(f"[WARN] H={H} oder W={W} nicht durch 8 teilbar (3x (1,2,2)-Pooling)")

def make_ds(X, Y, shuffle=True):
    """
    Creates a tensorflow dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds

print(">>> Phase 2: Create Tensorflow Datasets...")
train_ds = make_ds(X_train, Y_train, True)
val_ds   = make_ds(X_val,   Y_val,   False)
test_ds  = make_ds(X_test,  Y_test,  False)
print(">>> Datasets created")


# %%
# ========= Defining 3D-U-Net Architecture ========

def conv_block(x, filters, kernel_size=(3,3,3), padding="same", activation="relu"):
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
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

def ms_ssim_loss_sampled(y_true, y_pred, k=1):
    """
    Defining MS-SSIM for the loss function equivalently as in the paper
    """
    # y: (B, D, H, W, C)
    batch_size = tf.shape(y_true)[0]
    depth = tf.shape(y_true)[1]                               # Number of 2D slices
    idx = _sample_depth_indices(batch_size, depth, k=k)       # (B,k)
    # gathering chosen slices
    y_groundtruth = tf.gather(y_true, idx, batch_dims=1)      # (B,k,H,W,C)
    y_model = tf.gather(y_pred, idx, batch_dims=1)            # (B,k,H,W,C)
    # flatten to 2D pictures
    y_groundtruth_2 = tf.reshape(y_groundtruth, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    y_model_2 = tf.reshape(y_model, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    ms  = tf.image.ssim_multiscale(y_groundtruth_2, y_model_2, max_val=1.0)     # (B*k,)
    return 1.0 - tf.reduce_mean(ms)

def combined_loss(y_true, y_pred, k_slices=1):
    """
    Combining the loss composite of MAE and MS-SSIM
    (MAE stable and useful for strong signals --> Bragg peaks)
    (MS-SSIM focuses on structure --> CDW satellite signals)
    """
    l_mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ms  = ms_ssim_loss_sampled(y_true, y_pred, k=k_slices)  # K=1
    return (1.0 - ALPHA) * l_mae + ALPHA * l_ms

def ms_ssim_metric(y_true, y_pred):
    """
    Showing MS-SSIM metric during training
    """
    yt2 = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], tf.shape(y_true)[4]))
    yp2 = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], tf.shape(y_pred)[4]))
    yt2 = tf.cast(yt2, tf.float32)
    yp2 = tf.cast(yp2, tf.float32)
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))

def psnr_metric(y_true, y_pred):
    """
    Showing PSNR metric during training
    """
    return tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)



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
      2) schreibt JSON mit Timestamp
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
            jsons = list(self.root.glob(base_stem + "_*.json"))
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
# ======== Train =======

print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=combined_loss,
    metrics=["mae", "mse", psnr_metric, ms_ssim_metric],
    jit_compile=False # Would be false per default, but just to be sure
)
# model.summary()

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor":"val_loss","patience":10},
    "data_prep": {"use_vst": False, "size": 5, "group_len": 41, "dtype": "float32"},
    "ALPHA": ALPHA,
}

bf = BestFinalizeCallback(ckpt_root, run_meta=run_meta, code_name="AUTO")  # AUTO nimmt Skriptnamen als Prefix

ckpt_best = callbacks.ModelCheckpoint(
    filepath=str(bf.tmp_path),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

cbs = [
    ckpt_best, bf,
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=2
)
print(">>> Training complete")

