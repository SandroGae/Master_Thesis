import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pathlib import Path

# ================= KONFIGURATION =================
ROOT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis")
ORIGINAL_DIR = ROOT_DIR / "original_data"
# Pfad wo die interpolierten Daten liegen (falls abweichend, hier anpassen)
DATA_DIR = ORIGINAL_DIR 

# Output Pfad
VIDEO_DIR = ORIGINAL_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Dateinamen
FILE_ORIG       = "test_data.hdf5"
FILE_INTERP_OFF = "interpolated_test_data_pois_off.hdf5"
FILE_INTERP_ON  = "interpolated_test_data_pois_on.hdf5"

# Parameter
SERIES_TO_PLOT = 12
LEN_ORIG       = 41

# --- KORREKTUR HIER ---
# Laut deinem Generator-Code ist N_interpolate = 5
N_INTERPOLATE  = 5  
# ----------------------

# Berechnung der Gesamtlänge: 
# Bei N=5 und Length=41: 41 + (40 * 5) = 241 Bilder pro Serie
LEN_INTERP     = LEN_ORIG + (LEN_ORIG - 1) * N_INTERPOLATE 
STEP_SIZE      = N_INTERPOLATE + 1 

FPS = 3
OUTPUT_FORMAT = "mp4"

# ================= HILFSFUNKTIONEN =================

def load_series(base_path, filename, series_idx_1based, length):
    """Lädt eine spezifische Serie sicher aus einem HDF5 File."""
    full_path = base_path / filename
    idx_0based = series_idx_1based - 1
    start = idx_0based * length
    end   = start + length
    
    print(f"-> Lade: {filename} (Serie {series_idx_1based})...")
    print(f"   Lese Index {start} bis {end} (Länge: {length})")
    
    # Pfad-Fallback Logik
    if not full_path.exists():
        fallback = base_path / "interpolated_data_linear" / filename
        if fallback.exists():
            full_path = fallback
        else:
            print(f"❌ DATEI NICHT GEFUNDEN: {full_path}")
            return None, None

    try:
        with h5py.File(full_path, 'r') as f:
            low  = f['low_count/data'][:, :, start:end]
            high = f['high_count/data'][:, :, start:end]
            return low, high
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return None, None

# ================= 1. DATEN LADEN =================

print("Starte Datenimport...")
low_orig, high_orig = load_series(ORIGINAL_DIR, FILE_ORIG, SERIES_TO_PLOT, LEN_ORIG)
low_int_off, high_int_off = load_series(ORIGINAL_DIR, FILE_INTERP_OFF, SERIES_TO_PLOT, LEN_INTERP)
low_int_on, high_int_on = load_series(ORIGINAL_DIR, FILE_INTERP_ON, SERIES_TO_PLOT, LEN_INTERP)

if any(x is None for x in [low_orig, low_int_off, low_int_on]):
    print("Abbruch: Dateien fehlen.")
    exit()

# ================= 2. NORMALISIERUNG (STRATEGIE: REFERENCE ORIGINAL) =================
# Strategie: Damit das Original "gut" aussieht (wie im ersten Video), berechnen wir
# die Percentile NUR auf dem Original. Wir wenden diese Grenzen dann auf ALLE an.
# Das garantiert Vergleichbarkeit UND guten Kontrast für das Original.

print("Berechne Skalierung basierend auf Originaldaten (0.5% - 99.5%)...")

# Low Count Skala
vmin_l, vmax_l = np.percentile(low_orig, [0.5, 99.5])
# High Count Skala
vmin_h, vmax_h = np.percentile(high_orig, [0.5, 99.5])

print(f"Skala Low:  {vmin_l:.4f} bis {vmax_l:.4f}")
print(f"Skala High: {vmin_h:.4f} bis {vmax_h:.4f}")

# Erstelle die Norm-Objekte
norm_l = Normalize(vmin=vmin_l, vmax=vmax_l)
norm_h = Normalize(vmin=vmin_h, vmax=vmax_h)

# ================= 3. PLOT SETUP =================
fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
fig.suptitle(f'Vergleich Serie {SERIES_TO_PLOT} (Ref: Original Skala)', fontsize=16)

# Setup Datenlisten
datasets_l = [low_orig, low_int_off, low_int_on]
datasets_h = [high_orig, high_int_off, high_int_on]
titles     = ["Original", "Interpolated (Clean)", "Interpolated (Poisson)"]

artists_l = []
artists_h = []

for i in range(3):
    # LOW
    ax_l = axes[i, 0]
    ax_l.set_title(f"{titles[i]} (Low)")
    # WICHTIG: Alle nutzen norm_l
    img_l = ax_l.imshow(datasets_l[i][:,:,0], cmap='gray_r', norm=norm_l, origin='lower')
    ax_l.axis('off')
    artists_l.append(img_l)
    
    # HIGH
    ax_h = axes[i, 1]
    ax_h.set_title(f"{titles[i]} (High)")
    # WICHTIG: Alle nutzen norm_h
    img_h = ax_h.imshow(datasets_h[i][:,:,0], cmap='gray_r', norm=norm_h, origin='lower')
    ax_h.axis('off')
    artists_h.append(img_h)

status_text = fig.text(0.5, 0.02, 'Init...', ha='center', fontsize=12, fontweight='bold')

# ================= 4. ANIMATION LOOP =================
def update(frame_idx):
    # Berechnung des Index für das Original (Step-Funktion)
    # Wenn N=5, dann haben wir 5 Zwischenbilder. 
    # Index-Mapping: 0->0, 1..5->0, 6->1
    orig_idx = min(frame_idx // STEP_SIZE, LEN_ORIG - 1)
    
    # Update Zeile 1: Original
    artists_l[0].set_data(low_orig[:, :, orig_idx])
    artists_h[0].set_data(high_orig[:, :, orig_idx])
    
    # Update Zeile 2: Clean Interp
    artists_l[1].set_data(low_int_off[:, :, frame_idx])
    artists_h[1].set_data(high_int_off[:, :, frame_idx])
    
    # Update Zeile 3: Noise Interp
    artists_l[2].set_data(low_int_on[:, :, frame_idx])
    artists_h[2].set_data(high_int_on[:, :, frame_idx])
    
    # Status Text
    is_orig = (frame_idx % STEP_SIZE == 0)
    tag = "ORIGINAL" if is_orig else "INTERP"
    status_text.set_text(f"Frame {frame_idx}/{LEN_INTERP-1} (Orig: {orig_idx}) | {tag}")
    
    return artists_l + artists_h + [status_text]

# Start Animation
print(f"Rendere Video ({LEN_INTERP} Frames)...")
ani = FuncAnimation(fig, update, frames=LEN_INTERP, interval=1000/FPS, blit=False)

# Speichern
out_name = f"comparison_final_serie_{SERIES_TO_PLOT}.{OUTPUT_FORMAT}"
out_path = VIDEO_DIR / out_name

print(f"Speichere nach: {out_path}")
try:
    if OUTPUT_FORMAT == "mp4":
        ani.save(str(out_path), writer='ffmpeg', fps=FPS)
    else:
        ani.save(str(out_path), writer='pillow', fps=FPS)
    print("✅ Fertig.")
except Exception as e:
    print(f"❌ Fehler: {e}")

plt.close(fig)