import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------
# PATHS (keine Umbenennung: wir ueberschreiben exakt dieselben Dateinamen)
# ------------------------------------------------------------
NPY_PATH = Path(r"C:\Users\sandr\VS_Master_Thesis\DANMAX\data_2.npy")
OUT_DIR = NPY_PATH.parent

P_LOW  = 0.5
P_HIGH = 99.5

OUT_YG  = OUT_DIR / f"ct_yellowgreen_p{P_LOW}-p{P_HIGH}_fullres.png"
OUT_VIR = OUT_DIR / f"ct_viridis_p{P_LOW}-p{P_HIGH}_fullres.png"

# ------------------------------------------------------------
# KONTRAST-FIX (viel besser als nur vmin/vmax):
# 1) robustes Windowing (Percentiles)
# 2) asinh-Kompression (holt Details raus)
# 3) CLAHE (lokaler Kontrast, CT-typisch fuer "sieht gut aus")
# 4) optional gamma
#
# Hinweis: CLAHE braucht opencv. Wenn nicht installiert -> faellt sauber zurueck.
# ------------------------------------------------------------
ASINH_ALPHA = 14.0   # 8..25
GAMMA = 0.85         # <1 mehr "Punch" in dunklen Bereichen

USE_CLAHE = True
CLAHE_CLIP = 2.0     # 1.5..3.0
CLAHE_TILE = (8, 8)  # (8,8) oder (16,16)

# Anzeige-Settings (nur fürs Fenster)
FIG_DPI  = 300
FIG_SIZE = (7, 7)

# Gelb -> Gruen (ohne blau)
YG_CMAP = LinearSegmentedColormap.from_list(
    "yellow_green_custom",
    ["#fff59d", "#dce775", "#aed581", "#66bb6a", "#2e7d32"]
)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
img = np.load(NPY_PATH)
if img.ndim != 2:
    raise ValueError(f"Expected 2D array, got shape {img.shape}")

print("Loaded:", img.shape, img.dtype)

# ------------------------------------------------------------
# PERCENTILE WINDOW
# ------------------------------------------------------------
vmin, vmax = np.percentile(img, (P_LOW, P_HIGH))
if vmax <= vmin:
    vmax = vmin + 1e-12

print(f"Percentile window: p{P_LOW}={vmin:.6g}, p{P_HIGH}={vmax:.6g}")

# Normalize to [0,1] with clipping
x = (img - vmin) / (vmax - vmin)
x = np.clip(x, 0.0, 1.0)

# asinh compression (global contrast shaping)
x = np.arcsinh(ASINH_ALPHA * x) / np.arcsinh(ASINH_ALPHA)
x = np.clip(x, 0.0, 1.0)

# optional CLAHE (local contrast, makes CT look "right")
if USE_CLAHE:
    try:
        import cv2
        x8 = (x * 255.0 + 0.5).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        x8 = clahe.apply(x8)
        x = x8.astype(np.float32) / 255.0
        x = np.clip(x, 0.0, 1.0)
        print("CLAHE: on")
    except Exception as e:
        print("CLAHE: off (opencv not available) ->", repr(e))

# gamma
x = np.clip(x, 0.0, 1.0) ** GAMMA

# ------------------------------------------------------------
# EXPORT-FIX (dein Fehler kam von 16-bit RGB PNG via Pillow/ImageIO)
# -> Wir speichern stattdessen:
#    A) 16-bit GRAYSCALE PNG (sieht extrem sauber aus)  [PIL kann das]
#    B) 8-bit FARB-PNG mit gelb->gruen colormap (sieht gut aus, aber 8-bit)
#
# WICHTIG: Dateinamen bleiben IDENTISCH, wir ueberschreiben.
# ------------------------------------------------------------
def save_16bit_grayscale_png(norm01, out_path):
    # 16-bit single-channel PNG: beste "Quali" (keine Banding-Artefakte)
    try:
        from PIL import Image
        g16 = (np.clip(norm01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        im = Image.fromarray(g16, mode="I;16")
        im.save(str(out_path))
        print("Saved 16-bit grayscale:", out_path)
        return True
    except Exception as e:
        print("16-bit grayscale save failed ->", repr(e))
        return False

def save_8bit_color_png(norm01, cmap, out_path):
    # 8-bit RGB PNG (lossless), farbig
    rgba = cmap(np.clip(norm01, 0.0, 1.0))
    rgb8 = (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb8, mode="RGB").save(str(out_path))
    except Exception:
        # Fallback
        plt.imsave(out_path, rgb8)
    print("Saved 8-bit color:", out_path)

# 1) ct_yellowgreen_...png: wir behalten den Namen, speichern aber farbig (8-bit)
save_8bit_color_png(x, YG_CMAP, OUT_YG)

# 2) ct_viridis_...png: ueberschreiben mit 16-bit GRAYSCALE (beste Qualitaet)
#    Falls du UNBEDINGT farbig willst, kommentiere diese Zeile aus und nutze save_8bit_color_png.
ok = save_16bit_grayscale_png(x, OUT_VIR)
if not ok:
    save_8bit_color_png(x, plt.get_cmap("viridis"), OUT_VIR)

# ------------------------------------------------------------
# PREVIEW
# ------------------------------------------------------------
plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
plt.imshow(x, cmap=YG_CMAP, interpolation="nearest")
plt.axis("off")
plt.title(f"Preview | p{P_LOW}-{P_HIGH} | asinh={ASINH_ALPHA} | gamma={GAMMA}")
plt.tight_layout()
plt.show()

print("Done. Files overwritten (same names) in:", OUT_DIR)

# Tuning (ohne Dateinamen zu aendern):
# - Mehr Kontrast: ASINH_ALPHA=20, GAMMA=0.75, CLAHE_CLIP=2.5
# - Weniger aggressiv: ASINH_ALPHA=10, GAMMA=0.95, CLAHE_CLIP=1.5
