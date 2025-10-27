from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === EXAKT EINE der beiden Zeilen aktiv lassen ===
csv_path = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\csv_combined_2D.csv")  # ReLU-Runs
# csv_path = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\csv_combined_3D.csv")    # ELU-Runs

# Hidden-Aktivierung + Tag aus Dateinamen ableiten (Fallbacks)
name_lower = csv_path.name.lower()
if "relu" in name_lower or "2d" in name_lower:
    HIDDEN_ACT = "ReLU"
    RUNSET_TAG = "2D"
elif "elu" in name_lower or "3d" in name_lower:
    HIDDEN_ACT = "ReLU"
    RUNSET_TAG = "3D"
else:
    HIDDEN_ACT = "unknown"
    RUNSET_TAG = "unk"

# Wenn du die Hidden-Aktivierung erzwingen willst:
# HIDDEN_ACT = "ReLU"  # oder "ELU"

# Mapping fuer Output-Activation (output layer)
ACT_OUT_MAP = {1.0: "sigmoid", 2.0: "tanh", 3.0: "linear"}

# Output-Verzeichnis (KEINE Unterordner)
out_dir = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\plots_unet_3d_JENS_V2_test")
out_dir.mkdir(parents=True, exist_ok=True)

# CSV laden
df = pd.read_csv(csv_path).dropna(how="all")

# Erwartete Spalten (ohne ssim/val_ssim)
expected = [
    "epoch","base_filters","batch_size","depth","loss","lr","mae","mse",
    "out_act_id","psnr","run_tag_hash","val_loss","val_mae","val_mse","val_psnr"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten in CSV: {missing}")

# Run-Reihenfolge (aeltester -> neuester)
run_order = df.dropna(subset=["run_tag_hash"])["run_tag_hash"].drop_duplicates().tolist()

# Epochen auf 1..20 (statt 0..19)
df["epoch_1based"] = df["epoch"].astype(int) + 1
x_ticks = list(range(1, 21))

def get_meta(g: pd.DataFrame):
    d  = int(g["depth"].iloc[0])        if pd.notna(g["depth"].iloc[0])        else -1
    bf = int(g["base_filters"].iloc[0]) if pd.notna(g["base_filters"].iloc[0]) else -1
    bs = int(g["batch_size"].iloc[0])   if pd.notna(g["batch_size"].iloc[0])   else -1
    lr = str(g["lr"].iloc[0])
    actid_raw = g["out_act_id"].iloc[0]
    actid = float(actid_raw) if pd.notna(actid_raw) else np.nan
    out_actname = ACT_OUT_MAP.get(actid, "unknown")
    return d, bf, bs, lr, actid, out_actname

def add_legends(ax, depth, bf, bs, lr, out_actname, hidden_act):
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, 0.98),
              ncol=2, frameon=True, fontsize=8)
    param_text = (
        f"Depth: {depth}\n"
        f"Base filters: {bf}\n"
        f"Batch size: {bs}\n"
        f"LR: {lr}\n"
        f"Activation function: {hidden_act}\n"
        f"Output Activation: {out_actname}"
    )
    ax.text(0.98, 0.98, param_text,
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# Plotten (nur individuelle y-Skalen)
for idx, run_hash in enumerate(run_order, start=1):
    g = df[df["run_tag_hash"] == run_hash].copy().sort_values("epoch_1based")
    depth, bf, bs, lr, actid, out_actname = get_meta(g)

    fig, ax = plt.subplots()
    ax.plot(g["epoch_1based"], g["loss"],     label="Training loss")
    ax.plot(g["epoch_1based"], g["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Sweep Run {idx}")
    ax.set_xticks(x_ticks)
    ax.grid(True, which="both", linestyle="--", alpha=1)
    add_legends(ax, depth, bf, bs, lr, out_actname, HIDDEN_ACT)
    fig.tight_layout()

    fname = (
        f"{idx:02d}_{RUNSET_TAG}_d{depth}_bf{bf}_bs{bs}_lr{lr}_"
        f"outAct{int(actid) if np.isfinite(actid) else 'NA'}_loss.png"
    )
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)

print(f"Fertig. Plots liegen in:\n  {out_dir}")
