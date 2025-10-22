import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CSV laden ---
csv = r"C:\Users\sandr\VS_Master_Thesis\Plots\unet_3d_JENS_V2_combined.csv"
df = pd.read_csv(csv)

# --- bis zu 36 Runs identifizieren ---
run_order = df["run_tag_hash"].dropna().drop_duplicates().tolist()[:36]

# --- Farbpalette (36 Farben) ---
palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
colors = palette[:len(run_order)]

# --- Output-Ordner: eine Ebene höher ./plots_scout ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
out_dir = script_dir.parent / "Plots" / "plots_scout"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Y-Achsen-Intervalle pro Plot ---
ylims = {
    "val_loss": (0.15, 0.35),
    "val_mae":  (0.00, 0.15),
    "val_mse":  (0.00, 0.03),
}

def plot_and_save(metric_col, title, ylabel, filename):
    plt.figure()
    for i, run_id in enumerate(run_order):
        g = df[df["run_tag_hash"] == run_id].sort_values("epoch")
        plt.plot(g["epoch"], g[metric_col], linewidth=1.0, color=colors[i], label=f"Run {i+1}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    if metric_col in ylims:
        ymin, ymax = ylims[metric_col]
        plt.ylim(ymin, ymax)
    plt.tight_layout()
    stem = Path(filename).stem
    plt.savefig(out_dir / f"{stem}_rescaled.png", dpi=200)
    plt.close()

# 1) Validation loss
plot_and_save("val_loss", "Validation Loss over Epochs (36 runs)", "val_loss",
              "val_loss_all_runs.png")

# 2) Validation MAE
plot_and_save("val_mae", "Validation MAE over Epochs (36 runs)", "val_mae",
              "val_mae_all_runs.png")

# 3) Validation MSE
plot_and_save("val_mse", "Validation MSE over Epochs (36 runs)", "val_mse",
              "val_mse_all_runs.png")

print(f"Saved plots to: {out_dir}")
