import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pathlib import Path

print("start")
# Ordnerstruktur
ROOT_DIR = Path(r"C:\Users\sandr\VS_Master_Thesis")
ORIGINAL_DIR = ROOT_DIR / "original_data"
DATA_DIR = ORIGINAL_DIR
VIDEO_DIR = ORIGINAL_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Dateinamen
FILE_ORIG       = "test_data.hdf5"
FILE_INTERP_OFF = "interpolated_test_data_pois_off.hdf5"
FILE_INTERP_ON  = "interpolated_test_data_pois_on.hdf5"

# Parameter
SERIES_TO_PLOT = 12
LEN_ORIG       = 41
N_INTERPOLATE  = 5  # Anzahl Bilder interpoliert zwischen zwei Bildern
FPS = 3
OUTPUT_FORMAT = "mp4"
LEN_INTERP     = LEN_ORIG + (LEN_ORIG - 1) * N_INTERPOLATE # Länger der Serien (Aktuell 241 für N=5)
STEP_SIZE      = N_INTERPOLATE + 1


def load_series(base_path, filename, series_idx_1based, length):
    """
    Lädt eine spezifische Serie direkt aus dem angegebenen Pfad.
    """
    full_path = base_path / filename
    idx_0based = series_idx_1based - 1 # Index berechnen
    start = idx_0based * length
    end   = start + length

    with h5py.File(full_path, 'r') as f:
        low  = f['low_count/data'][:, :, start:end]
        high = f['high_count/data'][:, :, start:end]

        return low, high
           
low_orig, high_orig = load_series(ORIGINAL_DIR, FILE_ORIG, SERIES_TO_PLOT, LEN_ORIG)
low_int_off, high_int_off = load_series(ORIGINAL_DIR, FILE_INTERP_OFF, SERIES_TO_PLOT, LEN_INTERP)
low_int_on, high_int_on = load_series(ORIGINAL_DIR, FILE_INTERP_ON, SERIES_TO_PLOT, LEN_INTERP)


# Normalisierung durch Berechnung der Percentile auf dem Originalbild und Anwenundung auf alle
vmin_l, vmax_l = np.percentile(low_orig, [0.5, 99.5]) # Low Count Skala
vmin_h, vmax_h = np.percentile(high_orig, [0.5, 99.5]) # High Count Skala

norm_l = Normalize(vmin=vmin_l, vmax=vmax_l) # Erstelle die Norm-Objekte
norm_h = Normalize(vmin=vmin_h, vmax=vmax_h)

# Plot
fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
fig.suptitle(f'Vergleich Serie {SERIES_TO_PLOT} (Ref: Original Skala)', fontsize=16)
datasets_l = [low_orig, low_int_off, low_int_on]
datasets_h = [high_orig, high_int_off, high_int_on]
titles     = ["Original", "Interpolated (Clean)", "Interpolated (Poisson)"]
artists_l = []
artists_h = []

for i in range(3):
    ax_l = axes[i, 0]
    ax_l.set_title(f"{titles[i]} (Low)")
    img_l = ax_l.imshow(datasets_l[i][:,:,0], cmap='gray_r', norm=norm_l, origin='lower')
    ax_l.axis('off')
    artists_l.append(img_l)
    ax_h = axes[i, 1]
    ax_h.set_title(f"{titles[i]} (High)")
    img_h = ax_h.imshow(datasets_h[i][:,:,0], cmap='gray_r', norm=norm_h, origin='lower')
    ax_h.axis('off')
    artists_h.append(img_h)

status_text = fig.text(0.5, 0.02, 'Init...', ha='center', fontsize=12, fontweight='bold')

# Animation loop
def update(frame_idx):
    orig_idx = min(frame_idx // STEP_SIZE, LEN_ORIG - 1)
    artists_l[0].set_data(low_orig[:, :, orig_idx])
    artists_h[0].set_data(high_orig[:, :, orig_idx])
    artists_l[1].set_data(low_int_off[:, :, frame_idx])
    artists_h[1].set_data(high_int_off[:, :, frame_idx])
    artists_l[2].set_data(low_int_on[:, :, frame_idx])
    artists_h[2].set_data(high_int_on[:, :, frame_idx])
   
    # Status
    is_orig = (frame_idx % STEP_SIZE == 0)
    tag = "ORIGINAL" if is_orig else "INTERP"
    status_text.set_text(f"Frame {frame_idx}/{LEN_INTERP-1} (Orig: {orig_idx}) | {tag}")
   
    return artists_l + artists_h + [status_text]

ani = FuncAnimation(fig, update, frames=LEN_INTERP, interval=1000/FPS, blit=False) # Start Animation

out_name = f"comparison_serie_{SERIES_TO_PLOT}.{OUTPUT_FORMAT}" # Speichern
out_path = VIDEO_DIR / out_name
ani.save(str(out_path), writer='ffmpeg', fps=FPS)
plt.close(fig)
print("done")