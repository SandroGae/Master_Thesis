import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CSV laden ---
csv = r"C:\Users\sandr\VS_Master_Thesis\Plots\csv_combined_robust.csv"
df = pd.read_csv(csv)

# --- bis zu 36 Runs identifizieren ---
run_order = df["run_tag_hash"].dropna().drop_duplicates().tolist()[:36]

# --- Farbpalette (36 Farben) ---
palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
colors = palette[:len(run_order)]

# --- Output-Ordner ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
out_dir = script_dir.parent / "Plots" / "plots_scout"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Y-Limits fuer rescaled ---
ylims = {
    "val_loss": (0.04, 0.15),
    "val_mae":  (0.00, 0.08),
    "val_mse":  (0.00, 0.01),
}

def _plot(metric_col, title, ylabel, apply_rescale: bool, show_legend: bool):
    plt.figure()
    for i, run_id in enumerate(run_order):
        g = df[df["run_tag_hash"] == run_id].sort_values("epoch")
        plt.plot(g["epoch"], g[metric_col], linewidth=1.0, color=colors[i], label=f"Run {i+1}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    if apply_rescale and metric_col in ylims:
        ymin, ymax = ylims[metric_col]
        plt.ylim(ymin, ymax)
    if show_legend:
        # oben rechts, 4 Spalten, kompakt
        plt.legend(
            loc="upper right", bbox_to_anchor=(0.98, 0.98),
            ncol=4, fontsize=7, frameon=True, fancybox=True, framealpha=0.9,
            borderpad=0.3, labelspacing=0.3, handlelength=1.5, handletextpad=0.4
        )
    plt.tight_layout()


def plot_both(metric_col, title, ylabel, filename_base):
    # 1) unskaliert MIT Legende (oben rechts, 4 Spalten)
    _plot(metric_col, title, ylabel, apply_rescale=False, show_legend=True)
    plt.savefig(out_dir / f"{filename_base}_robust.png", dpi=200)
    plt.close()

    # 2) rescaled OHNE Legende
    _plot(metric_col, title, ylabel, apply_rescale=True, show_legend=False)
    plt.savefig(out_dir / f"{filename_base}_rescaled_robust.png", dpi=200)
    plt.close()

# Plots erstellen
plot_both("val_loss", "Validation Loss over Epochs (36 runs)", "val_loss", "val_loss_all_runs")
plot_both("val_mae",  "Validation MAE over Epochs (36 runs)",  "val_mae",  "val_mae_all_runs")
plot_both("val_mse",  "Validation MSE over Epochs (36 runs)",  "val_mse",  "val_mse_all_runs")

print(f"Saved plots to: {out_dir}")
