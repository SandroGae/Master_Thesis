import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import matplotlib.patheffects as pe

# =====================================================
# 1) SETUP
# =====================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# CSV liegt im selben Ordner wie das Skript
CSV_FILE = SCRIPT_DIR.parent / "Evaluation_Metrics_H_K_L" / "Full_Evaluation_Results.csv"

# Optional: absoluter Pfad (falls du willst)
# CSV_FILE = Path(r"C:\Users\sandr\VS_Master_Thesis\Unet\Analysis_ROI\codes\Evaluation_Metrics_H_K_L\Full_Evaluation_Results.csv")

OUT_PNG = SCRIPT_DIR / "Heatmaps_Interpolated_All_vs_Clean.png"
OUT_PDF = SCRIPT_DIR / "Heatmaps_Interpolated_All_vs_Clean.pdf"

# 7x7 Grid in 1/6-Schritten
GRID = np.round(np.linspace(0, 1, 7), 4)
EXPECTED_SEEDS_PER_CELL = 10  # wie von dir beschrieben

# =====================================================
# 2) HILFSFUNKTIONEN
# =====================================================
def parse_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}

def snap_to_sixth(x):
    """Mappt z.B. 0.1667 sauber auf 1/6."""
    try:
        v = float(x)
    except Exception:
        return np.nan
    v = np.clip(v, 0.0, 1.0)
    return float(np.round(v * 6) / 6)

def sem(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(n))

def interpolate_surface(df_points, z_col, n_grid=220):
    """
    Interpoliert eine Flaeche aus den Aggregationspunkten.
    Fallback: NaNs aus linearer Interpolation werden mit nearest gefuellt.
    """
    x = df_points["Alpha"].values
    y = df_points["Beta"].values
    z = df_points[z_col].values

    xi = np.linspace(0, 1, n_grid)
    yi = np.linspace(0, 1, n_grid)
    XI, YI = np.meshgrid(xi, yi)

    ZI_lin = griddata((x, y), z, (XI, YI), method="linear")
    ZI_near = griddata((x, y), z, (XI, YI), method="nearest")

    if ZI_lin is None and ZI_near is None:
        raise RuntimeError(f"Interpolation fehlgeschlagen fuer {z_col}")

    if ZI_lin is None:
        ZI = ZI_near
    else:
        ZI = ZI_lin.copy()
        if ZI_near is not None:
            mask = ~np.isfinite(ZI)
            ZI[mask] = ZI_near[mask]

    return XI, YI, ZI

def robust_norm(a, b):
    vals = []
    for arr in [a, b]:
        if arr is None:
            continue
        flat = np.asarray(arr).ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat):
            vals.append(flat)
    if not vals:
        return Normalize(vmin=0, vmax=1)
    vals = np.concatenate(vals)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if np.isclose(vmin, vmax):
        pad = 1e-6 if vmin == 0 else abs(vmin) * 0.01
        vmin -= pad
        vmax += pad
    return Normalize(vmin=vmin, vmax=vmax)

def add_alpha1_expansion(df):
    """
    Falls fuer Alpha=1.0 nur ein Beta-Wert existiert, dupliziere auf alle 7 Betas.
    """
    a1_mask = np.isclose(df["Alpha"].values, 1.0, atol=1e-8)
    if not a1_mask.any():
        return df

    df_a1 = df.loc[a1_mask].copy()
    unique_b = np.sort(df_a1["Beta"].unique())

    if len(unique_b) <= 1:
        base_b = unique_b[0] if len(unique_b) == 1 else 0.0
        extra = []
        for b in GRID:
            if np.isclose(b, base_b):
                continue
            tmp = df_a1.copy()
            tmp["Beta"] = float(b)
            extra.append(tmp)
        if extra:
            df = pd.concat([df] + extra, ignore_index=True)
            print("Alpha=1.0 Rand wurde auf alle 7 Beta-Werte erweitert.")
    return df

def draw_panel(ax, df_points, z_col, title, cmap, norm,
               show_xlabel=False, show_ylabel=False,
               value_fmt="{:.3f}", uncertainty_df=None,
               show_point_labels=False):
    """
    uncertainty_df (nur fuer clean-Panels) erwartet Spalten:
    Alpha, Beta, NSeedsClean, NSeedsTotal, SEM
    """
    # Interpolierte Flaeche
    XI, YI, ZI = interpolate_surface(df_points, z_col)
    levels = 60

    cf = ax.contourf(XI, YI, ZI, levels=levels, cmap=cmap, norm=norm)
    ax.contour(XI, YI, ZI, levels=14, colors="white", alpha=0.35, linewidths=0.6)

    # Sample-Punkte
    ax.scatter(df_points["Alpha"], df_points["Beta"],
               s=28, c="white", edgecolors="black", linewidths=0.6, alpha=0.75, zorder=4)

    # Clean-Unsicherheit: Whisker werden laenger wenn Seeds fehlen
    if uncertainty_df is not None and len(uncertainty_df) > 0:
        u = uncertainty_df.copy()

        # Optional: Marker-Groesse leicht nach SEM
        sem_col = "SEM"
        sem_vals = u[sem_col].values if sem_col in u.columns else np.zeros(len(u))
        finite_sem = sem_vals[np.isfinite(sem_vals)]
        sem_min = float(np.min(finite_sem)) if len(finite_sem) else 0.0
        sem_max = float(np.max(finite_sem)) if len(finite_sem) else 1.0
        sem_span = sem_max - sem_min if sem_max > sem_min else 1.0

        for _, r in u.iterrows():
            x = float(r["Alpha"])
            y = float(r["Beta"])
            n_clean = int(r["NSeedsClean"]) if pd.notna(r["NSeedsClean"]) else 0
            n_total = int(r["NSeedsTotal"]) if pd.notna(r["NSeedsTotal"]) else EXPECTED_SEEDS_PER_CELL
            missing = max(0, n_total - n_clean)
            frac_missing = missing / max(1, n_total)

            # Whisker-Laenge in Achsenkoordinaten
            half_len = 0.012 + 0.035 * frac_missing
            cap = 0.006 + 0.008 * frac_missing
            lw = 0.9 + 1.1 * frac_missing

            # schwarzer vertikaler Whisker
            ax.plot([x, x], [y - half_len, y + half_len], color="black", lw=lw, alpha=0.95, zorder=5)
            ax.plot([x - cap, x + cap], [y - half_len, y - half_len], color="black", lw=lw, alpha=0.95, zorder=5)
            ax.plot([x - cap, x + cap], [y + half_len, y + half_len], color="black", lw=lw, alpha=0.95, zorder=5)

            # Ring-Groesse leicht nach SEM
            s_sem = 45.0
            if pd.notna(r.get("SEM", np.nan)):
                s_sem = 35 + 60 * ((float(r["SEM"]) - sem_min) / sem_span)

            # Ring um den Punkt
            ax.scatter([x], [y], s=s_sem, facecolors="none", edgecolors="black",
                       linewidths=0.8, alpha=0.7, zorder=5)

    # Werte an Punkten (optional)
    if show_point_labels:
        for _, r in df_points.iterrows():
            txt = value_fmt.format(float(r[z_col]))
            t = ax.text(float(r["Alpha"]), float(r["Beta"]), txt,
                        fontsize=7, color="white", ha="center", va="center", zorder=6)
            t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black", alpha=0.9)])

    # Achsenformat
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(GRID)
    ax.set_yticks(GRID)
    ax.set_xticklabels([f"{g:.3f}" for g in GRID], rotation=45, ha="right")
    ax.set_yticklabels([f"{g:.3f}" for g in GRID])

    ax.set_title(title, fontsize=13, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Alpha", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Beta", fontsize=11)

    # dezente Achsen
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

    return cf

# =====================================================
# 3) CSV LADEN & BEREINIGEN
# =====================================================
if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV nicht gefunden:\n{CSV_FILE}")

df_raw = pd.read_csv(CSV_FILE, comment="#", skip_blank_lines=True)

# Spalten robust finden
colmap = {c.lower().replace("_", "").replace(" ", ""): c for c in df_raw.columns}

required = ["alpha", "beta", "seed", "arearatio", "posshift"]
for key in required:
    if key not in colmap:
        raise ValueError(f"Spalte '{key}' fehlt. Gefundene Spalten: {list(df_raw.columns)}")

# Clean-Spalte finden (IsClean oder Is_Clean)
clean_col = None
for cand in ["isclean", "iscleaned", "clean", "isok"]:
    if cand in colmap:
        clean_col = colmap[cand]
        break

df = pd.DataFrame({
    "Alpha": pd.to_numeric(df_raw[colmap["alpha"]], errors="coerce"),
    "Beta": pd.to_numeric(df_raw[colmap["beta"]], errors="coerce"),
    "Seed": pd.to_numeric(df_raw[colmap["seed"]], errors="coerce"),
    "AreaRatio": pd.to_numeric(df_raw[colmap["arearatio"]], errors="coerce"),
    "PosShift": pd.to_numeric(df_raw[colmap["posshift"]], errors="coerce"),
})

if clean_col is not None:
    df["IsClean"] = df_raw[clean_col].map(parse_bool)
else:
    df["IsClean"] = True
    print("Keine IsClean/Is_Clean-Spalte gefunden -> alle Seeds werden als clean behandelt.")

df = df.dropna(subset=["Alpha", "Beta", "Seed", "AreaRatio", "PosShift"]).copy()
df["Seed"] = df["Seed"].astype(int)

# Auf 1/6-Grid snappen
df["Alpha"] = df["Alpha"].map(snap_to_sixth)
df["Beta"] = df["Beta"].map(snap_to_sixth)
df = df.dropna(subset=["Alpha", "Beta"]).copy()

# Alpha=1 Rand erweitern (Quadrat-Topologie)
df = add_alpha1_expansion(df)

# Nur erwartetes Grid behalten
df = df[df["Alpha"].isin(GRID) & df["Beta"].isin(GRID)].copy()

# =====================================================
# 4) AGGREGATIONEN
# =====================================================
# 4a) Alle Runs (wirklich Mittel ueber alle Zeilen = Seeds * Serien)
all_runs = (
    df.groupby(["Alpha", "Beta"], as_index=False)
      .agg(
          AreaRatio=("AreaRatio", "mean"),
          PosShift=("PosShift", "mean"),
          NRows=("Seed", "size")
      )
)

# 4b) Seed-Level (wichtig fuer clean-Statistik und n_clean)
seed_level = (
    df.groupby(["Alpha", "Beta", "Seed"], as_index=False)
      .agg(
          AreaSeed=("AreaRatio", "mean"),
          ShiftSeed=("PosShift", "mean"),
          IsCleanSeed=("IsClean", "all")
      )
)

seed_counts = (
    seed_level.groupby(["Alpha", "Beta"], as_index=False)
             .agg(NSeedsTotal=("Seed", "nunique"))
)

clean_seed_stats = (
    seed_level[seed_level["IsCleanSeed"]]
    .groupby(["Alpha", "Beta"], as_index=False)
    .agg(
        AreaRatio=("AreaSeed", "mean"),   # Mittel ueber clean Seeds
        PosShift=("ShiftSeed", "mean"),   # Mittel ueber clean Seeds
        NSeedsClean=("Seed", "nunique"),
        AreaSEM=("AreaSeed", sem),
        ShiftSEM=("ShiftSeed", sem)
    )
)

clean_points = seed_counts.merge(clean_seed_stats, on=["Alpha", "Beta"], how="left")

# Falls keine clean Seeds in einer Zelle
clean_points["NSeedsClean"] = clean_points["NSeedsClean"].fillna(0)
clean_points["AreaSEM"] = clean_points["AreaSEM"].fillna(np.nan)
clean_points["ShiftSEM"] = clean_points["ShiftSEM"].fillna(np.nan)

# Nur Zellen mit mindestens 1 clean Seed koennen geplottet/interpoliert werden
clean_points_plot = clean_points[clean_points["NSeedsClean"] > 0].copy()

# =====================================================
# 5) INTERPOLATIONEN FUER NORMS (gleiche Skala links/rechts)
# =====================================================
_, _, Z_area_all = interpolate_surface(all_runs, "AreaRatio")
_, _, Z_area_clean = interpolate_surface(clean_points_plot.rename(columns={"AreaRatio": "AreaRatio"}), "AreaRatio")

_, _, Z_shift_all = interpolate_surface(all_runs, "PosShift")
_, _, Z_shift_clean = interpolate_surface(clean_points_plot.rename(columns={"PosShift": "PosShift"}), "PosShift")

norm_area = robust_norm(Z_area_all, Z_area_clean)
norm_shift = robust_norm(Z_shift_all, Z_shift_clean)

# =====================================================
# 6) PLOTTING (Optik analog zu deinem Referenzstil)
# =====================================================
fig = plt.figure(figsize=(18, 12), dpi=150)
gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10)

# Oben links: AreaRatio ALL
ax00 = fig.add_subplot(gs[0, 0])
cf00 = draw_panel(
    ax=ax00,
    df_points=all_runs,
    z_col="AreaRatio",
    title="Normalized σ·A (GT=1) — All runs",
    cmap="plasma",
    norm=norm_area,
    show_xlabel=False,
    show_ylabel=True,
    value_fmt="{:.3f}",
    uncertainty_df=None,
    show_point_labels=False
)

# Oben rechts: AreaRatio CLEAN
ax01 = fig.add_subplot(gs[0, 1])
unc_area = clean_points_plot[["Alpha", "Beta", "NSeedsClean", "NSeedsTotal", "AreaSEM"]].rename(columns={"AreaSEM": "SEM"})
cf01 = draw_panel(
    ax=ax01,
    df_points=clean_points_plot[["Alpha", "Beta", "AreaRatio"]].copy(),
    z_col="AreaRatio",
    title="Normalized σ·A (GT=1) — Clean seeds only",
    cmap="plasma",
    norm=norm_area,
    show_xlabel=False,
    show_ylabel=False,
    value_fmt="{:.3f}",
    uncertainty_df=unc_area,
    show_point_labels=False
)

# Unten links: PosShift ALL
ax10 = fig.add_subplot(gs[1, 0])
cf10 = draw_panel(
    ax=ax10,
    df_points=all_runs,
    z_col="PosShift",
    title="Average peak shift |Δμ| vs GT — All runs",
    cmap="magma_r",
    norm=norm_shift,
    show_xlabel=True,
    show_ylabel=True,
    value_fmt="{:.2f}",
    uncertainty_df=None,
    show_point_labels=False
)

# Unten rechts: PosShift CLEAN
ax11 = fig.add_subplot(gs[1, 1])
unc_shift = clean_points_plot[["Alpha", "Beta", "NSeedsClean", "NSeedsTotal", "ShiftSEM"]].rename(columns={"ShiftSEM": "SEM"})
cf11 = draw_panel(
    ax=ax11,
    df_points=clean_points_plot[["Alpha", "Beta", "PosShift"]].copy(),
    z_col="PosShift",
    title="Average peak shift |Δμ| vs GT — Clean seeds only",
    cmap="magma_r",
    norm=norm_shift,
    show_xlabel=True,
    show_ylabel=False,
    value_fmt="{:.2f}",
    uncertainty_df=unc_shift,
    show_point_labels=False
)

# Colorbars je Zeile (gemeinsam)
cbar_top = fig.colorbar(cf00, ax=[ax00, ax01], fraction=0.02, pad=0.02)
cbar_top.set_label("Mean AreaRatio = (A·σ)_pred / (A·σ)_gt", fontsize=10)

cbar_bottom = fig.colorbar(cf10, ax=[ax10, ax11], fraction=0.02, pad=0.02)
cbar_bottom.set_label("Mean peak shift |Δμ| relative to GT", fontsize=10)

# Titel + Hinweis
fig.suptitle(
    "Hyperparameter Heatmaps (7x7 grid)\n"
    "Top: normalized Gaussian area proxy (AreaRatio), Bottom: average peak shift |Δμ|",
    fontsize=15, fontweight="bold", y=0.98
)

fig.text(
    0.5, 0.015,
    "Clean-panels: black whiskers become longer when more seeds are missing (10 - n_clean). "
    "White points mark the original 7x7 sample locations.",
    ha="center", va="bottom", fontsize=9
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(OUT_PNG, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

# =====================================================
# 7) DEBUG / KONTROLLAUSGABE
# =====================================================
print("\n" + "=" * 90)
print(f"CSV: {CSV_FILE}")
print(f"Zeilen nach Cleaning: {len(df)}")
print(f"Alpha unique: {sorted(df['Alpha'].unique())}")
print(f"Beta unique:  {sorted(df['Beta'].unique())}")
print("-" * 90)

# Kontrolle pro Zelle
diag = seed_counts.merge(clean_points[["Alpha", "Beta", "NSeedsClean"]], on=["Alpha", "Beta"], how="left")
diag["NSeedsClean"] = diag["NSeedsClean"].fillna(0).astype(int)
diag["Missing"] = diag["NSeedsTotal"] - diag["NSeedsClean"]
diag = diag.sort_values(["Alpha", "Beta"])
print(diag.head(20).to_string(index=False))

print("-" * 90)
print(f"Gespeichert:\n  {OUT_PNG}\n  {OUT_PDF}")
print("=" * 90)