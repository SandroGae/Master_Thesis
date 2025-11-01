#!/usr/bin/env python3
# normalisation_check_2d.py – Histogramm-Analyse (gesamt + Einzelbilder)

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
import tensorflow as tf
tf.config.optimizer.set_jit(False)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from jens_stuff import SumScaleNormalizer, reset_random_seeds
from train_utils import build_1stack_datasets_flat, clip01

# ----------------------------------------------------------
# Setup
# ----------------------------------------------------------
reset_random_seeds(0)
BATCH_SIZE = 8
N_IMAGES   = 400
N_BINS     = 200
N_BINS_pictures = 30

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "original_data"
OUT_DIR  = ROOT / "data" / "analysis_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Normalisierung (wie Jens)
# ----------------------------------------------------------
preproc_train = SumScaleNormalizer(scale_min=5000,  scale_max=15000, normalize_label=True, batch_mode=False)
preproc_val   = SumScaleNormalizer(scale_min=10000, scale_max=10001, normalize_label=True, batch_mode=False)

def pipeline_train(x, y):
    x, y = preproc_train.map(x, y)
    return clip01(x), clip01(y)

def pipeline_val(x, y):
    x, y = preproc_val.map(x, y)
    return clip01(x), clip01(y)

# ----------------------------------------------------------
# Lade Datensätze (RAW + normalisiert)
# ----------------------------------------------------------
def _id(x, y): return x, y

train_raw, val_raw, _, _ = build_1stack_datasets_flat(
    data_dir=DATA_DIR,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    read_block=128,
    preproc_train=_id,
    preproc_eval=_id,
    out_rank=4,
    cache_after_preproc=False,
)

train_ds, val_ds, _, _ = build_1stack_datasets_flat(
    data_dir=DATA_DIR,
    batch_train=BATCH_SIZE,
    batch_eval=BATCH_SIZE,
    read_block=128,
    preproc_train=pipeline_train,
    preproc_eval=pipeline_val,
    out_rank=4,
    cache_after_preproc=False,
)

# ----------------------------------------------------------
# Sammle 400 Bilder (Low & High)
# ----------------------------------------------------------
def collect_images(ds, n):
    xs, ys = [], []
    for x, y in ds.unbatch().take(n):
        xs.append(x.numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.float32))
    return np.array(xs), np.array(ys)

print("Lese 400 Bilder...")
xtr_raw, ytr_raw = collect_images(train_raw, N_IMAGES)
xva_raw, yva_raw = collect_images(val_raw,   N_IMAGES)
xtr_norm, ytr_norm = collect_images(train_ds, N_IMAGES)
xva_norm, yva_norm = collect_images(val_ds,   N_IMAGES)

# ----------------------------------------------------------
# Min/Max von jedem 50. Bild (nur Kontrolle)
# ----------------------------------------------------------
def print_minmax_every_50(data, name):
    print(f"\n{name}")
    for i in range(0, len(data), 50):
        arr = data[i]
        print(f"Bild {i:03d}: min={arr.min():.3f}, max={arr.max():.3f}")

for arr, name in [
    (xtr_raw,  "Train LOW RAW"),  (ytr_raw,  "Train HIGH RAW"),
    (xtr_norm, "Train LOW norm"), (ytr_norm, "Train HIGH norm"),
    (xva_raw,  "Val LOW RAW"),    (yva_raw,  "Val HIGH RAW"),
    (xva_norm, "Val LOW norm"),   (yva_norm, "Val HIGH norm"),
]:
    print_minmax_every_50(arr, name)

# ----------------------------------------------------------
# Plot 1: Histogramme über alle Bilder (8 Subplots)
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
datasets = [
    (xtr_raw,  "Train LOW RAW"),
    (xtr_norm, "Train LOW norm"),
    (ytr_raw,  "Train HIGH RAW"),
    (ytr_norm, "Train HIGH norm"),
    (xva_raw,  "Val LOW RAW"),
    (xva_norm, "Val LOW norm"),
    (yva_raw,  "Val HIGH RAW"),
    (yva_norm, "Val HIGH norm"),
]

xmax_values = [110, 1, 1500, 1, 110, 1, 1500, 1]  # manuell definierbar

for ax, (data, title), xmax in zip(axes.ravel(), datasets, xmax_values):
    vals = data.ravel()
    ax.hist(vals, bins=N_BINS, range=(0, xmax), density=True, histtype='step')
    ax.set_title(f"{title}\n(xmax={xmax})")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Intensität")
    ax.set_ylabel("Dichte")
    ax.grid(alpha=0.3)

fig.suptitle(f"Histogramme (Train/Val × Low/High × RAW/norm) – {N_IMAGES} Bilder, {N_BINS} Bins")
out_path = OUT_DIR / f"hist_8subplots_{N_IMAGES}bilder_{N_BINS}bins_manual_xmax.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"[OK] Hauptplot gespeichert: {out_path}")

# ----------------------------------------------------------
# Plot 2: Einzelplots (jedes 50. Bild, jeweils 8 Subplots)
# Peak zentriert, x-Achse ab 0
# ----------------------------------------------------------
print("\nErzeuge 8-Subplot-Figuren für jedes 50. Bild (Peak zentriert)...")

indices = list(range(0, N_IMAGES, 50))

for idx in indices:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, (data, title) in zip(axes.ravel(), datasets):
        vals = data[idx].ravel()

        # Erst grob Peak finden (über gesamten Bereich)
        hist_full, edges_full = np.histogram(vals, bins=N_BINS_pictures * 3, density=True)
        centers_full = 0.5 * (edges_full[:-1] + edges_full[1:])
        peak_x = centers_full[np.argmax(hist_full)]

        # Bereich auf Peak-basiertes Fenster einschränken
        x_min = 0
        x_max = 2 * peak_x if peak_x > 0 else edges_full[-1]

        # Jetzt Histogramm nur im sichtbaren Bereich neu berechnen
        hist, edges = np.histogram(vals, bins=N_BINS_pictures, range=(x_min, x_max), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])


        ax.plot(centers, hist, color="blue")
        ax.axvline(peak_x, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{title}\n(Bild {idx}, Peak={peak_x:.2f})", fontsize=10)
        ax.set_xlabel("Intensität")
        ax.set_ylabel("Dichte")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Histogramme für Bild {idx} (Peak zentriert, 8 Subplots)", fontsize=14)
    fname = OUT_DIR / f"hist_subplots_Bild{idx:03d}_peak_centered.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)

print(f"[OK] Alle Einzelplots mit 8 Subplots gespeichert in: {OUT_DIR}")
