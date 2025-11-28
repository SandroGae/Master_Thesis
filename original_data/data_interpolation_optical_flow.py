import h5py
import numpy as np
from pathlib import Path
import cv2

# Knfiguration
USE_POISSON_NOISE = True
N_INTERPOLATE = 5  # Zwischenbilder
FLOW_METHOD = "tvl1" 

# WICHTIG: Hier die gewünschte Serie einstellen
TARGET_SERIES_IDX = 12 
SERIES_LENGTH = 41     # Länge einer Serie in den Originaldaten

def apply_poisson_noise(image_data):
    clean_data = np.maximum(image_data, 0)
    noisy_data = np.random.poisson(clean_data).astype(np.float32)
    return noisy_data

def get_optical_flow(prev, next, method='farneback'):
    return cv2.calcOpticalFlowFarneback(prev,   
                                        next, 
                                        None, 
                                        0.5,    # Zoomfaktor Pyramide
                                        3,      # Stufen der Pyramide
                                        20,     # Fenstergrösse
                                        3,      # Anzahl der Iterationen pro Stufe
                                        5,      # Grösse des Pixel-Blocks
                                        1.2,    # Standardabweichung
                                        0       # Flags
)

def warp_image(img, flow):
    h, w = img.shape
    flow_map = np.column_stack(np.meshgrid(np.arange(w), np.arange(h)))
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
    map_x = flow_map[:,:,0] - flow[:,:,0]
    map_y = flow_map[:,:,1] - flow[:,:,1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def process_single_series(group_name, input_h5, output_h5, series_idx):
    """
    Verarbeitet NUR EINE spezifische Serie, um Zeit zu sparen.
    """
    # 1. Indizes berechnen
    start_idx = series_idx * SERIES_LENGTH
    end_idx   = start_idx + SERIES_LENGTH
    
    print(f" -> Lade '{group_name}' | Serie {series_idx} (Bilder {start_idx} bis {end_idx})...")
    
    # 2. Nur diesen Teil laden (Slicing direkt von der Festplatte)
    # HDF5 Shape ist (H, W, N), wir slicen N
    data_chunk = input_h5[f"{group_name}/data"][:, :, start_idx:end_idx]
    
    # Transpose zu (N, H, W) für Iteration
    image_series = data_chunk.transpose(2, 0, 1)
    
    N_current, Height, Width = image_series.shape
    
    # 3. Neue Größe berechnen
    new_series_len = SERIES_LENGTH + (SERIES_LENGTH - 1) * N_INTERPOLATE
    
    # Output Array
    new_data = np.zeros((new_series_len, Height, Width), dtype=np.float32)
    current_idx = 0
    
    # 4. Startbild speichern
    new_data[current_idx] = image_series[0]
    current_idx += 1
    
    # 5. Interpolation Loop
    for i in range(SERIES_LENGTH - 1):
        img_a = image_series[i]
        img_b = image_series[i+1]
        
        # Flow berechnen
        flow = get_optical_flow(img_a, img_b, method=FLOW_METHOD)
        
        # Zwischenbilder generieren
        for j in range(1, N_INTERPOLATE + 1):
            t = j / (N_INTERPOLATE + 1)
            
            flow_a = flow * t
            flow_b = flow * (t - 1.0)
            
            warped_a = warp_image(img_a, flow_a)
            warped_b = warp_image(img_b, flow_b)
            
            img_interp = (1 - t) * warped_a + t * warped_b
            
            if USE_POISSON_NOISE:
                img_final = apply_poisson_noise(img_interp)
            else:
                img_final = img_interp
            
            new_data[current_idx] = img_final
            current_idx += 1
        
        # Originalbild B speichern
        new_data[current_idx] = img_b
        current_idx += 1
        
    print(f"Fertig. Neue Länge: {new_series_len} Frames.")

    # Speichern
    if group_name not in output_h5:
        group = output_h5.create_group(group_name)
    else:
        group = output_h5[group_name]
        
    # Zurück transponieren zu (H, W, N)
    final_data = new_data.transpose(1, 2, 0)
    group.create_dataset("data", data=final_data, compression="gzip")

if __name__ == "__main__":
    # Pfade
    ROOT_DIR = Path.home() / "data"
    IN_DIR   = ROOT_DIR / "original_data"
    OUT_DIR  = ROOT_DIR / "interpolated_data_optical_flow"
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Test-Data
    input_filename = "test_data.hdf5"
    
    # Deteiname
    suffix = "_flow_pois_on" if USE_POISSON_NOISE else "_flow_pois_off"
    output_filename = f"interpolated_test_data_S{TARGET_SERIES_IDX}{suffix}.hdf5"
    
    input_path  = IN_DIR / input_filename
    output_path = OUT_DIR / output_filename
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Extrahiere NUR Serie {TARGET_SERIES_IDX}...")
    

    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for key in ['high_count', 'low_count']:
            process_single_series(key, f_in, f_out, TARGET_SERIES_IDX)