import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pathlib import Path

print("Start Visualisierung (Synchronisierte Skalierung)...")

# ================= KONFIGURATION =================
ROOT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis")
ORIGINAL_DIR = ROOT_DIR / "original_data"
VIDEO_DIR = ORIGINAL_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 1. Die Datei, die wir VISUALISIEREN wollen (Flow)
FILE_FLOW = "interpolated_test_data_S12_flow_pois_on.hdf5"

# 2. Die Datei, von der wir die HELLIGKEIT (Skala) klauen wollen
FILE_ORIG = "test_data.hdf5" 

# Parameter (müssen zur Serie passen)
SERIES_IDX_ORIG = 12 
LEN_ORIG = 41
FPS = 5
OUTPUT_FORMAT = "mp4"
OUTPUT_VIDEO_NAME = f"vis_FIXED_{FILE_FLOW.replace('.hdf5', '')}.{OUTPUT_FORMAT}"

# ================= DATEN LADEN =================

def load_reference_scale(path, series_idx, length):
    """
    Lädt NUR die Original-Serie 12, um vmin/vmax zu berechnen.
    Damit sieht das Video exakt so aus wie im Vergleichs-Code.
    """
    full_path = ORIGINAL_DIR / path
    if not full_path.exists():
        print(f"WARNUNG: Referenzdatei {path} nicht gefunden!")
        return None, None

    print(f"Lade Referenz-Skala aus: {path} (Serie {series_idx})...")
    # Index berechnen (1-based zu 0-based)
    start = (series_idx - 1) * length
    end   = start + length
    
    with h5py.File(full_path, 'r') as f:
        low_ref  = f['low_count/data'][:, :, start:end]
        high_ref = f['high_count/data'][:, :, start:end]
        
    # Berechne die Skala basierend auf dem Original (Exakt wie im Vergleichs-Code)
    vmin_l, vmax_l = np.percentile(low_ref, [0.5, 99.5])
    vmin_h, vmax_h = np.percentile(high_ref, [0.5, 99.5])
    
    return (vmin_l, vmax_l), (vmin_h, vmax_h)

def load_flow_data(filename):
    path = ORIGINAL_DIR / filename
    print(f"Lade Flow-Daten aus: {filename}")
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
        
    with h5py.File(path, 'r') as f:
        low = f['low_count/data'][:]
        high = f['high_count/data'][:]
    return low, high

# 1. Flow Daten laden (Das wollen wir sehen)
try:
    low_flow, high_flow = load_flow_data(FILE_FLOW)
    n_frames = low_flow.shape[2]
except Exception as e:
    print(f"Fehler beim Laden der Flow Datei: {e}")
    exit()

# 2. Referenz-Skala holen (Damit es identisch aussieht!)
scale_l, scale_h = load_reference_scale(FILE_ORIG, SERIES_IDX_ORIG, LEN_ORIG)

# ================= NORMALISIERUNG =================
print("Wende Normalisierung an...")

if scale_l is not None:
    # Option A: Wir nutzen die Original-Skala (Identischer Look)
    vmin_l, vmax_l = scale_l
    vmin_h, vmax_h = scale_h
    print(f" -> Nutze Referenz-Skala Low: {vmin_l:.2f} bis {vmax_l:.2f}")
else:
    # Option B: Fallback (Selbst-Skalierung, falls Original fehlt)
    print(" -> Nutze Selbst-Skalierung (Keine Referenz gefunden).")
    vmin_l, vmax_l = np.percentile(low_flow, [0.5, 99.5])
    vmin_h, vmax_h = np.percentile(high_flow, [0.5, 99.5])

norm_l = Normalize(vmin=vmin_l, vmax=vmax_l)
norm_h = Normalize(vmin=vmin_h, vmax=vmax_h)


# ================= PLOT SETUP =================
# 1 Zeile, 2 Spalten
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
fig.suptitle(f'Single File Visualisierung: {FILE_FLOW}', fontsize=11)

# Linke Seite: Low Count
ax_l = axes[0]
ax_l.set_title("Optical Flow + Poisson (Low)")
img_l = ax_l.imshow(low_flow[:,:,0], cmap='gray_r', norm=norm_l, origin='lower')
ax_l.axis('off')

# Rechte Seite: High Count
ax_h = axes[1]
ax_h.set_title("Optical Flow + Poisson (High)")
img_h = ax_h.imshow(high_flow[:,:,0], cmap='gray_r', norm=norm_h, origin='lower')
ax_h.axis('off')

status_text = fig.text(0.5, 0.05, 'Init...', ha='center', fontsize=12)

# ================= ANIMATION LOOP =================
def update(frame_idx):
    img_l.set_data(low_flow[:, :, frame_idx])
    img_h.set_data(high_flow[:, :, frame_idx])
    status_text.set_text(f"Frame {frame_idx + 1} / {n_frames}")
    return [img_l, img_h, status_text]

print(f"Starte Rendering ({n_frames} Frames)...")
# blit=False ist oft stabiler bei Skalen-Wechseln, hier egal, wir nehmen False wie im Original
ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/FPS, blit=False)

out_path = VIDEO_DIR / OUTPUT_VIDEO_NAME
print(f"Speichere Video nach: {out_path}")

try:
    if OUTPUT_FORMAT == "mp4":
        ani.save(str(out_path), writer='ffmpeg', fps=FPS)
    else:
        ani.save(str(out_path), writer='pillow', fps=FPS)
    print("Fertig.")
except Exception as e:
    print(f"FEHLER beim Speichern: {e}")

plt.close(fig)