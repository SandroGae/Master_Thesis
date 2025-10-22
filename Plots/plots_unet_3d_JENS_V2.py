from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Pfad zur kombinierten CSV anpassen, falls noetig ===
csv_path = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\unet_3d_JENS_V2_combined.csv")

# Output-Verzeichnisse
root_out = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\plots_unet_3d_JENS_V2")
out_indiv = root_out / "individual_y_scale"
out_same  = root_out / "same_y_scale"
out_indiv.mkdir(parents=True, exist_ok=True)
out_same.mkdir(parents=True, exist_ok=True)

# CSV laden
df = pd.read_csv(csv_path).dropna(how="all")
expected = ["epoch","base_filters","batch_size","depth","loss","lr","mae","mse",
            "out_act_id","psnr","run_tag_hash","ssim","val_loss","val_mae","val_mse","val_psnr","val_ssim"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten in CSV: {missing}")

# Run-Reihenfolge: aeltester -> neuester (Run 1..36)
run_order = df.dropna(subset=["run_tag_hash"])["run_tag_hash"].drop_duplicates().tolist()

# Epochen auf 1..20 (statt 0..19)
df["epoch_1based"] = df["epoch"].astype(int) + 1
x_ticks = list(range(1, 21))

# Globale y-Skala fuer same_y_scale
global_max = np.nanmax([df["loss"].to_numpy(), df["val_loss"].to_numpy()])
ymin_same, ymax_same = 0.0, float(global_max * 1.05 if np.isfinite(global_max) else 1.0)

# Mapping fuer Activation
ACT_MAP = {1.0: "sigmoid", 2.0: "tanh", 3.0: "linear"}

def get_meta(g):
    d  = int(g["depth"].iloc[0]) if pd.notna(g["depth"].iloc[0]) else -1
    bf = int(g["base_filters"].iloc[0]) if pd.notna(g["base_filters"].iloc[0]) else -1
    bs = int(g["batch_size"].iloc[0]) if pd.notna(g["batch_size"].iloc[0]) else -1
    lr = str(g["lr"].iloc[0])
    actid_raw = g["out_act_id"].iloc[0]
    actid = float(actid_raw) if pd.notna(actid_raw) else np.nan
    actname = ACT_MAP.get(actid, "unknown")
    return d, bf, bs, lr, actid, actname

def add_legends(ax, depth, bf, bs, lr, actname):
    # Legende oben mittig: Train/Val – kleinerer Font
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, 0.98),
              ncol=2, frameon=True, fontsize=8)

    param_text = (
        f"Depth: {depth}\n"
        f"Base filters: {bf}\n"
        f"Batch size: {bs}\n"
        f"LR: {lr}\n"
        f"Activation function: ELU\n"
        f"Output Activation: {actname}"
    )
    ax.text(
        0.98, 0.98, param_text,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )


for idx, run_hash in enumerate(run_order, start=1):
    g = df[df["run_tag_hash"] == run_hash].copy().sort_values("epoch_1based")
    depth, bf, bs, lr, actid, actname = get_meta(g)

    # ---- individuelle y-Skala ----
    fig, ax = plt.subplots()
    ax.plot(g["epoch_1based"], g["loss"],     label="Training loss")    # blau
    ax.plot(g["epoch_1based"], g["val_loss"], label="Validation loss")  # orange
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Sweep Run {idx}")
    ax.set_xticks(x_ticks)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    add_legends(ax, depth, bf, bs, lr, actname)
    fig.tight_layout()
    fname_indiv = f"{idx:02d}_d{depth}_bf{bf}_bs{bs}_lr{lr}_act{int(actid) if np.isfinite(actid) else 'NA'}_loss_individual.png"
    fig.savefig(out_indiv / fname_indiv, dpi=300)
    plt.close(fig)

    # ---- gleiche y-Skala ----
    fig, ax = plt.subplots()
    ax.plot(g["epoch_1based"], g["loss"],     label="Training loss")
    ax.plot(g["epoch_1based"], g["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Sweep Run {idx}")
    ax.set_xticks(x_ticks)
    ax.set_ylim(ymin_same, ymax_same)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    add_legends(ax, depth, bf, bs, lr, actname)
    fig.tight_layout()
    fname_same = f"{idx:02d}_d{depth}_bf{bf}_bs{bs}_lr{lr}_act{int(actid) if np.isfinite(actid) else 'NA'}_loss_same.png"
    fig.savefig(out_same / fname_same, dpi=300)
    plt.close(fig)

print(f"Fertig. Plots liegen in:\n  {out_indiv}\n  {out_same}")
