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

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") # Increases performance without loss of quality (calculations still done with float_32 precision)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from unet_3d_data import prepare_in_memory_5to5
from pathlib import Path

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
    use_vst=True, # With anscombe transform
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
EPOCHS     = 400

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

    outputs = layers.Conv3D(1, (1,1,1), dtype="float32", activation=None)(c6)  # linear activation
    return models.Model(inputs, outputs, name="3D_U-Net")


# %%
# =========== Defining Metrics for training ========

def inv_anscombe(z):
    z = tf.maximum(z, 1e-3)
    z2 = z / 2.0
    # Unbiased approx. (gekürzt) fuer Poisson
    y = z2**2 - 1.0/8.0 + 1.0/(4.0*z**2) - 11.0/(8.0*z**4)
    return tf.nn.relu(y)

def psnr_orig_metric(y_true_vst, y_pred_vst):
    y_true = tf.clip_by_value(inv_anscombe(y_true_vst), 0.0, 1.0)
    y_pred = tf.clip_by_value(inv_anscombe(y_pred_vst), 0.0, 1.0)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ms_ssim_orig_metric(y_true_vst, y_pred_vst):
    yt = inv_anscombe(y_true_vst)
    yp = inv_anscombe(y_pred_vst)
    yt2 = tf.reshape(yt, (-1, tf.shape(yt)[2], tf.shape(yt)[3], tf.shape(yt)[4]))
    yp2 = tf.reshape(yp, (-1, tf.shape(yp)[2], tf.shape(yp)[3], tf.shape(yp)[4]))
    return tf.reduce_mean(tf.image.ssim_multiscale(yt2, yp2, max_val=1.0))


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
    # JSON direkt mit gleichem Basenamen wie das Modell (ohne Zeit im Namen)
    json_path = model_path.with_suffix(".json")
    ts = _timestamp()  # nur im Inhalt, nicht im Dateinamen

    meta = {
        "timestamp": ts,
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
            # JSON mit gleichem Basenamen (ohne Zeit im Namen)
            base_json = path.with_suffix(".json")
            jsons = [base_json] if base_json.exists() else []


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

print(">>> Phase 3: GPU training starts now!")

model = unet3d(input_shape=INPUT_SHAPE, base_filters=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mae",
    metrics=["mae", "mse", psnr_orig_metric, ms_ssim_orig_metric],
    jit_compile=False # Would be false per default, but just to be sure
)
# model.summary()

ckpt_root = Path.home() / "data" / "checkpoints_3d_unet_anscombe"
run_meta = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping": {"monitor":"val_loss","patience":10},
    "data_prep": {"use_vst": True, "size": 5, "group_len": 41, "dtype": "float32"},
}

bf = BestFinalizeCallback(ckpt_root, run_meta=run_meta)
ckpt_best = callbacks.ModelCheckpoint(
    filepath=str(bf.tmp_path),   # feste TEMP-Datei
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
