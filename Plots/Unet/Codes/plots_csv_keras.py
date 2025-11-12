from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# === CSV-Pfad anpassen ===
BASE    = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\Unet")
CSV_DIR = BASE / "CSV"
FIG_DIR = BASE / "Figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)

FILE_2d = "unet_2d_simple_loss0.0135_val0.0142.csv"
FILE_3d = "unet_3d_simple_loss0.0131_val0.0144.csv"
FILE_2d_200 = "unet_2d_simple_loss0.0118_val0.0135_epochs200.csv"
FILE_VDSR = "VDSR_reconstruction_V2_loss0.0132_val0.0133.csv"
FILE_3d_ssim_old = "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-084215.csv"
FILE_3d_ssim = "unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006.csv"
csv_path = CSV_DIR / FILE_3d_ssim_old

num_epochs = 50

# Laden
df = pd.read_csv(csv_path)
required_cols = {"epoch", "loss", "val_loss"}
if not required_cols.issubset(df.columns):
    raise ValueError("CSV braucht Spalten: epoch, loss, val_loss")

# epoch numerisch und sortieren
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df = df.dropna(subset=["epoch"]).copy()
df["epoch"] = df["epoch"].astype(int)
df = df.sort_values("epoch")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["epoch"], df["loss"], label="train", alpha=0.9)
plt.plot(df["epoch"], df["val_loss"], label="val", alpha=0.9)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train-/Val-Loss")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

out_png = FIG_DIR / f"{csv_path.stem}_loss_plot_epochs{num_epochs}.png"

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

print(f"Fertig: {out_png}")
