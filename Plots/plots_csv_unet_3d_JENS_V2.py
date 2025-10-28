from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# === HIER NUR DEN CSV-PFAD ANPASSEN ===
csv_path = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\csv_combined_2d_new_clean.csv")

# Ausgabeverzeichnis
out_dir = csv_path.parent / "plots_scout"
out_dir.mkdir(parents=True, exist_ok=True)

# Namenszusatz aus Dateiname (ohne "csv_combined")
name_part = csv_path.stem.replace("csv_combined", "").strip("_- ") or "run"

# CSV laden
df = pd.read_csv(csv_path)
# epoch sicher numerisch
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.dropna(subset=["epoch"]).copy()
df["epoch"] = df["epoch"].astype(int)

# Annahme: jeder Sweep hat genau 20 Einträge (0..19)
E = 20
n_rows = len(df)
n_sweeps = n_rows // E

# Helfer: einzelnen Sweep zurückgeben
def get_sweep(i: int):
    s = i * E
    e = s + E
    g = df.iloc[s:e].reset_index(drop=True)
    return g

# Plot-Funktion für eine Metrik
def plot_metric(metric: str, train_col: str, val_col: str, y_label: str):
    cols = min(4, max(1, n_sweeps))
    rows = math.ceil(n_sweeps / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8*cols, 3.6*rows), squeeze=False)
    fig.suptitle(metric, fontsize=16)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n_sweeps:
            g = get_sweep(i)
            x = g["epoch"] + 1  # 1..20
            if train_col in g:
                ax.plot(x, g[train_col], "--", label="train")
            if val_col in g:
                ax.plot(x, g[val_col], "-",  label="val")
            ax.set_title(f"sweep {i+1}")
            ax.set_xlabel("epoch"); ax.set_ylabel(y_label)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = out_dir / f"sweep_{metric}_{name_part}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

# Drei Bilder erzeugen
plot_metric("val_loss", "loss", "val_loss", "Loss")
plot_metric("val_mae",  "mae",  "val_mae",  "MAE")
plot_metric("val_psnr", "psnr", "val_psnr", "PSNR")

print(f"fertig. bilder in: {out_dir}")
