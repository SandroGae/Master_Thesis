# ==============================
# 0) Imports & global setup (identisch, training entfernt)
# ==============================

import os
# Deaktiviere XLA (just in time compiler) aus Stabilitaetsgruenden
os.environ["TF_DISABLE_XLA"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
# Behebe "Failed to allocate scratch space errors (testet verschiedene Faltungsalgorithmen bei der ersten Iteration)"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# Etfernt hartes Workspace-Limit (512 MB war zu klein)
os.environ.pop("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", None)
# GPU alloziert nur benoetigte Menge an VRAM
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Weniger TF/C++-Spam (INFO+WARNING weg, ERROR bleibt)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
# Unterdrueckt nervige Warnungen im Log
tf.get_logger().setLevel("ERROR")
from absl import logging as absl_logging
# XLA/absl-errors unterdruecken
absl_logging.set_verbosity(absl_logging.FATAL)

from pathlib import Path
import math
import re
from datetime import datetime  # nicht noetig fuer Checks, aber neutral zu lassen

from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_5stack_datasets_grouped, clip01

# Reproduzierbarkeit
seed = 0
reset_random_seeds(seed)
AUTO = tf.data.AUTOTUNE  # tensorflow waehlt selbst wie viele elemente parallel geladen/verarbeitet werden


# ==============================
# 1–3) Daten-Streaming (kein RAM-Fullload) — 1:1 wie im Trainingsteil
# ==============================

# Normalisierung identisch zu Jens
preproc_train_slice = SumScaleNormalizer(
    scale_range=[5000, 15001],
    pre_offset=0.0,
    normalize_label=True,  # Gleiche Normalisierung fuer high count Bilder
    axis=(2, 3, 4),        # Summation über Height, Width, Channels
    batch_mode=True,       # Skalierung geschieht pro Sample im Batch (False = Skalierungsfaktor ueber ganzes Datenset geschaetzt)
    clip_before=[0., float("inf")],
    clip_after=[0., 1.]
)
preproc_valid_slice = SumScaleNormalizer(
    scale_range=[5000, 5001],
    pre_offset=0.0,
    normalize_label=True,
    axis=(2, 3, 4),
    batch_mode=True,
    clip_before=[0., float("inf")],
    clip_after=[0., 1.]
)

BATCH_SIZE = 32

def map_slice_wise(normalizer):
    def _finite01(t):
        # Sichert Robustheit gegen NaN/Inf
        t = tf.cast(t, tf.float32)  # alle Operationen in float32
        t = tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))  # NaN/Inf -> 0
        return tf.clip_by_value(t, 0.0, 1.0)  # Clip [0,1]
    def _fn(x, y):
        # Normalisierung pro Slice
        x_norm, y_norm = normalizer.map(x, y)
        return _finite01(x_norm), _finite01(y_norm)
    return _fn

def augment_5stack_flips(x, y):
    # Spiegelt x und y zufaellig links-rechts und oben-unten
    # returns: geflipptes Paar (x, y)
    do_lr = tf.random.uniform(()) < 0.5  # left-right (W)
    do_ud = tf.random.uniform(()) < 0.5  # up-down (H)
    def fliplr(t): return tf.reverse(t, axis=[2])  # W
    def flipud(t): return tf.reverse(t, axis=[1])  # H
    x = tf.cond(do_lr, lambda: fliplr(x), lambda: x)
    y = tf.cond(do_lr, lambda: fliplr(y), lambda: y)
    x = tf.cond(do_ud, lambda: flipud(x), lambda: x)
    y = tf.cond(do_ud, lambda: flipud(y), lambda: y)
    return x, y

def resolve_data_dir():
    new = Path(r"C:\Users\sandr\VS_Master_Thesis\original_data")
    old = Path.home() / "data" / "original_data"

    if new.is_dir():
        return new
    if old.is_dir():
        return old

print(">>> Phase 1: Baue Datensatz via train_utils...")
DATA_DIR = resolve_data_dir()
train_ds, val_ds, test_ds, meta = build_5stack_datasets_grouped(
    data_dir=DATA_DIR,
    group_len=41,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    preproc_train=map_slice_wise(preproc_train_slice),
    preproc_eval=map_slice_wise(preproc_valid_slice),
    augmenter=augment_5stack_flips,
    deterministic=False,          # False: Map und Prefetch duerfen in zufaelliger Reihenfolge arbeiten
    cache_after_preproc=False,    # False: Augmentation wird bei jeder Epoche neu berechnet
)

def _steps(meta, split, batch):
    # Berechnet Anzahl Batches pro Epoche = Anzahl Updates pro Epoche
    spp = meta["samples_per_group"]  # 37 Trainingsdaten pro 41er Gruppe
    groups = meta[f"{split}_groups"]
    total = groups * spp
    return math.ceil(total / batch)

INPUT_SHAPE = meta["input_shape"]
print(">>> Datasets created")
print(f">>> INPUT_SHAPE: {INPUT_SHAPE}, steps/train ~ {_steps(meta, 'train', BATCH_SIZE)}")



# ===== Plausibilitaetschecks nur für TRAIN und VAL =====

print("\n>>> Phase 2: Plausibilitaetschecks (nur TRAIN & VAL)")

# Identitaets-Preproc (roh, ohne Normalisierung), Augmentierung aus fuer Determinismus
def identity_preproc(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return x, y

# --- RAW-Datasets (ohne Normalisierung, ohne Augmentierung) ---
print(">>> Erzeuge RAW-Datensaetze (ohne Norm, ohne Augmentierung)...")
train_raw, val_raw, _, _ = build_5stack_datasets_grouped(
    data_dir=DATA_DIR,
    group_len=41,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    preproc_train=identity_preproc,
    preproc_eval=identity_preproc,
    augmenter=None,
    deterministic=True,
    cache_after_preproc=False
)

# --- NORM-Datasets (mit SumScaleNormalizer, ohne Augmentierung) ---
print(">>> Erzeuge NORM-Datensaetze (mit Norm, ohne Augmentierung)...")
train_norm, val_norm, _, _ = build_5stack_datasets_grouped(
    data_dir=DATA_DIR,
    group_len=41,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    preproc_train=map_slice_wise(preproc_train_slice),
    preproc_eval=map_slice_wise(preproc_valid_slice),
    augmenter=None,
    deterministic=True,
    cache_after_preproc=False
)

def _reduce_dataset_stats(ds, collect_first=256):
    acc = {
        "n": 0,
        "x_sum_sum": 0.0, "x_sum_min": float("inf"), "x_sum_max": float("-inf"),
        "x_val_min": float("inf"), "x_val_max": float("-inf"),
        "y_sum_sum": 0.0, "y_sum_min": float("inf"), "y_sum_max": float("-inf"),
        "y_val_min": float("inf"), "y_val_max": float("-inf"),
        "x_sum_samples": [], "y_sum_samples": []
    }
    for xb, yb in ds:
        xb = tf.cast(xb, tf.float32); yb = tf.cast(yb, tf.float32)
        acc["x_val_min"] = min(acc["x_val_min"], float(tf.reduce_min(xb).numpy()))
        acc["x_val_max"] = max(acc["x_val_max"], float(tf.reduce_max(xb).numpy()))
        acc["y_val_min"] = min(acc["y_val_min"], float(tf.reduce_min(yb).numpy()))
        acc["y_val_max"] = max(acc["y_val_max"], float(tf.reduce_max(yb).numpy()))
        x_sums = tf.reduce_sum(xb, axis=[1,2,3,4]); y_sums = tf.reduce_sum(yb, axis=[1,2,3,4])
        acc["x_sum_sum"] += float(tf.reduce_sum(x_sums).numpy())
        acc["y_sum_sum"] += float(tf.reduce_sum(y_sums).numpy())
        acc["x_sum_min"]  = min(acc["x_sum_min"], float(tf.reduce_min(x_sums).numpy()))
        acc["x_sum_max"]  = max(acc["x_sum_max"], float(tf.reduce_max(x_sums).numpy()))
        acc["y_sum_min"]  = min(acc["y_sum_min"], float(tf.reduce_min(y_sums).numpy()))
        acc["y_sum_max"]  = max(acc["y_sum_max"], float(tf.reduce_max(y_sums).numpy()))
        acc["n"] += int(xb.shape[0])
        if len(acc["x_sum_samples"]) < collect_first:
            need = collect_first - len(acc["x_sum_samples"])
            acc["x_sum_samples"] += list(x_sums.numpy()[:need])
            acc["y_sum_samples"] += list(y_sums.numpy()[:need])
    acc["x_sum_mean"] = acc["x_sum_sum"] / acc["n"] if acc["n"] else float("nan")
    acc["y_sum_mean"] = acc["y_sum_sum"] / acc["n"] if acc["n"] else float("nan")
    acc["x_in_01"] = (acc["x_val_min"] >= -1e-6) and (acc["x_val_max"] <= 1.0 + 1e-6)
    acc["y_in_01"] = (acc["y_val_min"] >= -1e-6) and (acc["y_val_max"] <= 1.0 + 1e-6)
    return acc

def _print_stats_table(title, acc_raw, acc_norm):
    f = lambda v: f"{v:.6g}" if isinstance(v,(int,float)) and not (v!=v) else str(v)
    print(f"\n--- {title} ---")
    print("X (Input):")
    print(f"  RAW : sum_min={f(acc_raw['x_sum_min'])}  sum_mean={f(acc_raw['x_sum_mean'])}  "
          f"sum_max={f(acc_raw['x_sum_max'])}  |  val_range=[{f(acc_raw['x_val_min'])}, {f(acc_raw['x_val_max'])}]")
    print(f"  NORM: sum_min={f(acc_norm['x_sum_min'])}  sum_mean={f(acc_norm['x_sum_mean'])}  "
          f"sum_max={f(acc_norm['x_sum_max'])}  |  val_range=[{f(acc_norm['x_val_min'])}, {f(acc_norm['x_val_max'])}]  "
          f"in_[0,1]?={'JA' if acc_norm['x_in_01'] else 'NEIN'}")
    print("Y (Label):")
    print(f"  RAW : sum_min={f(acc_raw['y_sum_min'])}  sum_mean={f(acc_raw['y_sum_mean'])}  "
          f"sum_max={f(acc_raw['y_sum_max'])}  |  val_range=[{f(acc_raw['y_val_min'])}, {f(acc_raw['y_val_max'])}]")
    print(f"  NORM: sum_min={f(acc_norm['y_sum_min'])}  sum_mean={f(acc_norm['y_sum_mean'])}  "
          f"sum_max={f(acc_norm['y_sum_max'])}  |  val_range=[{f(acc_norm['y_val_min'])}, {f(acc_norm['y_val_max'])}]  "
          f"in_[0,1]?={'JA' if acc_norm['y_in_01'] else 'NEIN'}")
    xs_raw = acc_raw["x_sum_samples"][:30]; xs_nrm = acc_norm["x_sum_samples"][:30]
    print("  Beispiel-Summen (X) RAW -> NORM (erste 30):")
    for i in range(min(len(xs_raw), len(xs_nrm))):
        print(f"    {i:02d}: {f(xs_raw[i])}  ->  {f(xs_nrm[i])}")

# --- TRAIN ---
print(">>> TRAIN-Statistiken (RAW)...")
train_raw_stats  = _reduce_dataset_stats(train_raw, collect_first=64)
print(">>> TRAIN-Statistiken (NORM)...")
train_norm_stats = _reduce_dataset_stats(train_norm, collect_first=64)
_print_stats_table("TRAIN – Summen & Wertebereiche (RAW vs NORM)", train_raw_stats, train_norm_stats)

# --- VAL ---
print("\n>>> VAL-Statistiken (RAW)...")
val_raw_stats  = _reduce_dataset_stats(val_raw, collect_first=64)
print(">>> VAL-Statistiken (NORM)...")
val_norm_stats = _reduce_dataset_stats(val_norm, collect_first=64)
_print_stats_table("VAL – Summen & Wertebereiche (RAW vs NORM)", val_raw_stats, val_norm_stats)

def _hard_warn(acc_norm, split):
    if not acc_norm["x_in_01"] or not acc_norm["y_in_01"]:
        print(f"\n[WARNUNG] {split}: Normalisierte Werte ausserhalb [0,1]. "
              f"X in [0,1]? {acc_norm['x_in_01']}, Y in [0,1]? {acc_norm['y_in_01']}. "
              "Pruefe scale_range / clip_before / clip_after.")
_hard_warn(train_norm_stats, "TRAIN")
_hard_warn(val_norm_stats,   "VAL")

print("\n>>> Plausibilitaetschecks abgeschlossen.")




# ERGÄNZUNG

print("\n>>> Suche kleinstes X-Sample (Summe ueber H,W,Stack,Channel) im TRAIN und gebe mittleres Frame aus (RAW & NORM)...")

min_sum = float("inf")
min_raw_frame = None
min_norm_frame = None
min_sample_idx = -1

global_idx = 0  # laufender Sample-Zaehler ueber alle Batches
for (xb_raw, _), (xb_norm, _) in zip(train_raw, train_norm):
    # Sicherstellen, dass beides float32 ist
    xb_raw = tf.cast(xb_raw, tf.float32)
    xb_norm = tf.cast(xb_norm, tf.float32)

    # Summen pro Sample ueber H(=1), W(=2), Stack(=3), Channel(=4)
    sums = tf.reduce_sum(xb_raw, axis=[1, 2, 3, 4]).numpy()

    for i, s in enumerate(sums):
        if s < min_sum:
            min_sum = float(s)
            # mittleres Frame des 5-Stacks (Index 2)
            mid = int(xb_raw.shape[3] // 2)  # sollte 2 sein
            raw_frame = xb_raw[i, :, :, mid, :]   # (H, W, 1) erwartet
            norm_frame = xb_norm[i, :, :, mid, :] # (H, W, 1) erwartet

            # Channel-Dimension entfernen -> (H, W)
            min_raw_frame = tf.squeeze(raw_frame, axis=-1).numpy()
            min_norm_frame = tf.squeeze(norm_frame, axis=-1).numpy()
            min_sample_idx = global_idx + i

    global_idx += int(xb_raw.shape[0])

print(f"Minimale Summe (RAW, ueber alle Dimensionen) = {min_sum:.6g} bei globalem Sample-Index {min_sample_idx}")
print("RAW (mittleres Frame) als Matrix:")
print(min_raw_frame)

print("\nNORM (mittleres Frame) als Matrix:")
print(min_norm_frame)
