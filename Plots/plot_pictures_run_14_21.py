# unet3d_dual_infer_viz_norm.py
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from skimage.metrics import structural_similarity as ssim

# ====== Projekt-Root in sys.path (damit jens_stuff/train_utils auffindbar sind) ======
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from jens_stuff import SumScaleNormalizer  # <- deine Klasse

# ====== Pfade/Parameter ======
DATA_DIR       = Path(r"C:\Users\sandr\VS_Master_Thesis\data\original_data")
H5_NAME        = "test_data.hdf5"
CHECKPOINT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis\Plots\plots_serie_11_frame_18")
OUT_DIR        = CHECKPOINT_DIR
SAVE_DPI       = 200

# --- exakte Modell-Dateinamen (die zwei von dir) ---
SWEEP1_FILE  = "unet_3d_JENS_V2_sweep1_d3_bf8_ELU_LN_outsigmoid_lr0_0003_bs32__mae_ssim_0_7_V2_valloss_4.575e-01_PSNR_11.6.keras"
SWEEP32_FILE = "unet_3d_JENS_V2_sweep32_d4_bf16_ELU_LN_outsigmoid_lr0_0001_bs128__mae_ssim_0_7_V1_valloss_6.600e-02_PSNR_30.1.keras"

# IRUNet-Referenz (0-basiert)
GROUP_LEN = 41
SERIES_LABEL_ZEROBASED = 11   # "serie_011"
FRAME_LABEL_ZEROBASED  = 18   # "frame_018"

# Modelle: Muster fuer Run 14 und 21
PATTERN_RUN14 = "sweep_d4_bf16_ELU_LN_outsigmoid_lr0_0001_bs32__mae_ssim_0_7*.keras"
PATTERN_RUN21 = "sweep_d3_bf8_ELU_LN_outsigmoid_lr3e-05_bs128__mae_ssim_0_7*.keras"

# ===== Utils =====
def find_model(glob_pattern: str) -> Path:
    cands = sorted(CHECKPOINT_DIR.glob(glob_pattern))
    if not cands:
        raise FileNotFoundError(f"Kein Modell gefunden fuer Muster: {glob_pattern}\nIn: {CHECKPOINT_DIR}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def load_hwN(fp: Path):
    with h5py.File(fp, "r") as f:
        high = f["/high_count/data"][:]  # (H,W,N)
        low  = f["/low_count/data"][:]   # (H,W,N)
    # -> (N,H,W,1)
    high = high.transpose(2,0,1)[..., None].astype(np.float32)
    low  = low.transpose(2,0,1)[..., None].astype(np.float32)
    return high, low

def make_5stack_from_series(low_all, high_all, *, series_id, frame_idx, group_len=41):
    """liefert (x5,y5) als (1,5,H,W,1) + zentrale Slices (1,H,W,1) + center=2"""
    assert 2 <= frame_idx <= 38, "5er-Stack braucht 2 Nachbarn links/rechts."
    start = series_id * group_len
    rel = np.array([frame_idx-2, frame_idx-1, frame_idx, frame_idx+1, frame_idx+2], dtype=int)
    if np.any(rel < 0) or np.any(rel >= group_len):
        raise ValueError("5-Stack passt nicht in die 41er Serie.")
    abs_idx = start + rel
    x5 = low_all [abs_idx][None, ...]
    y5 = high_all[abs_idx][None, ...]
    low_center_raw  = low_all [start + frame_idx][None, ...]
    high_center_raw = high_all[start + frame_idx][None, ...]
    return x5, y5, low_center_raw, high_center_raw, 2

def robust_minmax(img, p1=1, p99=99):
    vals = img[...,0].ravel()
    vmin, vmax = np.percentile(vals, (p1, p99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals) + 1e-6)
    return float(vmin), float(vmax)

def show_quad(img1, img2, img3, img4, *, labels, title=None, save_path=None, cmap="gray_r"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, img, ttl in zip(axes, (img1,img2,img3,img4), labels):
        vmin, vmax = robust_minmax(img, 1, 99)
        ax.imshow(img[...,0], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(ttl); ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

# ===== einfache Metriken im NORM-Raum =====
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def psnr(a, b, data_range=1.0):
    err = np.mean((a - b)**2)
    if err == 0: return float("inf")
    return 20.0 * np.log10(data_range / np.sqrt(err))

def main():
    # --- Daten ---
    h5_path = DATA_DIR / H5_NAME
    high_all, low_all = load_hwN(h5_path)

    # --- Modelle (Run14/Run21 via Muster) ---
    model14_path = find_model(PATTERN_RUN14)
    model21_path = find_model(PATTERN_RUN21)
    print(f"[Run14] Lade: {model14_path.name}")
    print(f"[Run21] Lade: {model21_path.name}")
    model14 = models.load_model(str(model14_path), compile=False)
    model21 = models.load_model(str(model21_path), compile=False)

    # --- 5er-Stack exakt wie IRUNet-Referenz ---
    x5_raw, y5_raw, low_c_raw, high_c_raw, center = make_5stack_from_series(
        low_all, high_all,
        series_id=SERIES_LABEL_ZEROBASED, frame_idx=FRAME_LABEL_ZEROBASED, group_len=GROUP_LEN
    )

    # --- Normalisieren (identisch zu deinem Training/Eval) ---
    normalizer = SumScaleNormalizer(
        scale_range=[5000, 5001],
        pre_offset=0.0,
        normalize_label=True,
        axis=None,
        clip_before=[0., float("inf")],
        clip_after=[0., 1.],
        batch_mode=True,            # (B,5,H,W,1) -> ueber HWC summieren
    )
    Xn, Yn = normalizer.map(tf.convert_to_tensor(x5_raw), tf.convert_to_tensor(y5_raw))
    Xn = Xn.numpy(); Yn = Yn.numpy()

    # --- Inferenz im NORM-Raum (Run14/Run21) ---
    pred14_n = model14(Xn, training=False).numpy()
    pred21_n = model21(Xn, training=False).numpy()

    # --- Zentrumsslices (normiert) ---
    c = center
    low_n_c  = Xn      [0, c, ..., 0]
    high_n_c = Yn      [0, c, ..., 0]
    p14_n_c  = pred14_n[0, c, ..., 0]
    p21_n_c  = pred21_n[0, c, ..., 0]

    # --- Metriken (normiert) fuer Run14/Run21 ---
    print("\n[NORM metrics vs. Yn] (center slice, data_range=1.0)")
    for name, pred in [("Run14", p14_n_c), ("Run21", p21_n_c)]:
        m_mae  = mae (pred, high_n_c)
        m_rmse = rmse(pred, high_n_c)
        m_psnr = psnr(pred, high_n_c, data_range=1.0)
        m_ssim = float(ssim(pred, high_n_c, data_range=1.0))
        print(f"{name}: MAE={m_mae:.4f}  RMSE={m_rmse:.4f}  PSNR={m_psnr:.2f} dB  SSIM={m_ssim:.4f}")

    # --- Visualisierung (normiert) Run14/Run21 (unveraendert) ---
    series_1b = SERIES_LABEL_ZEROBASED + 1
    frame_1b  = FRAME_LABEL_ZEROBASED + 1
    title     = f"Serie {series_1b:02d} – Frame {frame_1b:02d} (normalized)"
    save_path = OUT_DIR / f"serie_{series_1b:02d}_frame_{frame_1b:02d}_quad_norm.png"

    show_quad(low_n_c[...,None], high_n_c[...,None], p14_n_c[...,None], p21_n_c[...,None],
              labels=("Low (norm)", "High (norm)", "Run14 (norm)", "Run21 (norm)"),
              title=title, save_path=save_path)

    # ====== ZUSATZ: SWEEP32 vs SWEEP1 — einmal NORM, einmal DENORM (gleiche Reihenfolge) ======
    # Modelle laden
    sweep1_path  = CHECKPOINT_DIR / SWEEP1_FILE
    sweep32_path = CHECKPOINT_DIR / SWEEP32_FILE
    if not sweep1_path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {sweep1_path}")
    if not sweep32_path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {sweep32_path}")

    print(f"[SWEEP1] Lade:  {sweep1_path.name}")
    print(f"[SWEEP32] Lade: {sweep32_path.name}")
    model_sw1  = models.load_model(str(sweep1_path),  compile=False)
    model_sw32 = models.load_model(str(sweep32_path), compile=False)

    # Inferenz im NORM-Raum
    pred_sw1_n  = model_sw1(Xn,  training=False).numpy()
    pred_sw32_n = model_sw32(Xn, training=False).numpy()

    # Zentrumsslices (norm)
    p_sw1_norm_c  = pred_sw1_n [0, c, ..., 0]
    p_sw32_norm_c = pred_sw32_n[0, c, ..., 0]

    # === Denormalisierung: y_raw = y_norm * sum_label / scale ===
    scale_val = float(tf.reshape(normalizer._denorm_pars['scale'], ()).numpy())
    sum_label_center = float(np.maximum(np.sum(y5_raw[0, c, ...]), 1e-12))
    p_sw1_denorm_c  = p_sw1_norm_c  * (sum_label_center / scale_val)
    p_sw32_denorm_c = p_sw32_norm_c * (sum_label_center / scale_val)

    # ---- Figur A: NORM-Vergleich (Low, High, sweep32 norm, sweep1 norm) ----
    save_path_norm = OUT_DIR / f"serie_{series_1b:02d}_frame_{frame_1b:02d}_sweep32_sweep1_norm_quad.png"
    show_quad(
        low_n_c[...,None],                        # 1) Low (norm)
        high_n_c[...,None],                       # 2) High (norm)
        p_sw32_norm_c[...,None],                  # 3) sweep32 (norm)
        p_sw1_norm_c[...,None],                   # 4) sweep1 (norm)
        labels=("Low (norm)", "High (norm)", "sweep32 (norm)", "sweep1 (norm)"),
        title=f"Serie {series_1b:02d} – Frame {frame_1b:02d} – sweep32 vs sweep1 (norm)",
        save_path=save_path_norm
    )

    # ---- Figur B: DENORM-Vergleich (Low, High, sweep32 denorm, sweep1 denorm) ----
    save_path_denorm = OUT_DIR / f"serie_{series_1b:02d}_frame_{frame_1b:02d}_sweep32_sweep1_denorm_quad.png"
    show_quad(
        low_n_c[...,None],                        # 1) Low (norm) bleibt links als Referenz
        high_n_c[...,None],                       # 2) High (norm) bleibt daneben
        p_sw32_denorm_c[...,None],                # 3) sweep32 (denorm)
        p_sw1_denorm_c[...,None],                 # 4) sweep1 (denorm)
        labels=("Low (norm)", "High (norm)", "sweep32 (denorm)", "sweep1 (denorm)"),
        title=f"Serie {series_1b:02d} – Frame {frame_1b:02d} – sweep32 vs sweep1 (denorm)",
        save_path=save_path_denorm
    )

    print(f"\nFertig. Gespeichert:\n  {save_path}\n  {save_path_norm}\n  {save_path_denorm}")
    print("[SHAPES] Xn", Xn.shape, " pred14_n", pred14_n.shape, " pred21_n", pred21_n.shape)

if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.FATAL)
    except Exception:
        pass
    main()
