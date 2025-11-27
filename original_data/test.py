import h5py
import numpy as np
from pathlib import Path
import cv2

# ================= KONFIGURATION =================
USE_POISSON_NOISE = True
N_INTERPOLATE = 5  # 5 Zwischenbilder

# Wähle den Algorithmus: 
# "farneback" = Extrem schnell, gut für flüssige Bewegung
# "tvl1"      = Genauer bei Kanten, langsamer, braucht 'opencv-contrib-python'
FLOW_METHOD = "farneback" 

def apply_poisson_noise(image_data):
    """
    Simuliert Zählraten-Rauschen (Poisson)
    """
    clean_data = np.maximum(image_data, 0)
    # Bei sehr kleinen Werten ist Poisson instabil, daher Sicherheitshalber clip
    noisy_data = np.random.poisson(clean_data).astype(np.float32)
    return noisy_data

def get_optical_flow(prev, next, method='farneback'):
    """
    Berechnet den Optical Flow mit OpenCV (Viel schneller als Skimage).
    """
    # OpenCV erwartet uint8 oder float32. Wir nutzen float32.
    if method == 'farneback':
        # Parameter für Farneback (für wissenschaftliche Daten oft gut)
        # pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow
    
    elif method == 'tvl1':
        # Dual TV-L1 (benötigt opencv-contrib-python)
        try:
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
            # Parameter tunen falls nötig:
            optical_flow.setLambda(0.15)
            optical_flow.setNumberIterations(100) 
            flow = optical_flow.calc(prev, next, None)
            return flow
        except AttributeError:
            print("ACHTUNG: 'opencv-contrib-python' fehlt für TVL1. Nutze Farneback.")
            return cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def warp_image(img, flow):
    """
    Warpt ein Bild basierend auf dem Flow Vektor mittels cv2.remap (sehr schnell).
    """
    h, w = img.shape
    # Grid erstellen
    flow_map = np.column_stack(np.meshgrid(np.arange(w), np.arange(h)))
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)

    # Den Flow addieren: Woher kommt der Pixel?
    # Inverse Mapping: map_x = x - flow_x
    map_x = flow_map[:,:,0] - flow[:,:,0]
    map_y = flow_map[:,:,1] - flow[:,:,1]

    # Remap
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def process_dataset(group_name, input_h5, output_h5):
    dataset = input_h5[f"{group_name}/data"][:]
    Height, Width, N_total = dataset.shape
    series_length = 41
    N_series = N_total // series_length
    
    # Transpose zu (N, H, W) für einfachere Iteration
    data_transposed = dataset.transpose(2, 0, 1) 
    
    # Neue Größe berechnen
    # Pro Lücke (40 Stück) kommen N_INTERPOLATE Bilder dazu
    new_series_len = series_length + (series_length - 1) * N_INTERPOLATE
    N_total_new = N_series * new_series_len
    
    print(f" -> Verarbeite '{group_name}': {N_series} Serien. Neue Länge pro Serie: {new_series_len}")
    
    # Output Array
    new_data = np.zeros((N_total_new, Height, Width), dtype=np.float32)

    current_idx = 0
    
    for idx in range(N_series):
        start = idx * series_length
        end = start + series_length
        image_series = data_transposed[start:end]

        # 1. Startbild speichern
        new_data[current_idx] = image_series[0]
        current_idx += 1
        
        # Loop durch die Paare in der Serie
        for i in range(series_length - 1):
            img_a = image_series[i]
            img_b = image_series[i+1]
            
            # Flow berechnen (nur 1x pro Paar!)
            # Wir berechnen Flow A -> B
            flow = get_optical_flow(img_a, img_b, method=FLOW_METHOD)
            
            # N Zwischenbilder generieren
            for j in range(1, N_INTERPOLATE + 1):
                # Zeitfaktor t von 0 bis 1
                t = j / (N_INTERPOLATE + 1)
                
                # Wir schieben A in die Zukunft (+ t * flow)
                # Wir schieben B in die Vergangenheit (- (1-t) * flow)
                
                # Achtung: OpenCV remap braucht "woher der Pixel kommt".
                # Wenn wir A nach t warpen wollen, müssen wir wissen wo der Pixel in A war.
                # Approx: Pixel bei (x) in t kommt von (x - t*flow) in A.
                
                flow_a = flow * t
                flow_b = flow * (t - 1.0) # Negativer Flow für B
                
                # Warpen
                warped_a = warp_image(img_a, flow_a)
                warped_b = warp_image(img_b, flow_b)
                
                # Blenden
                img_interp = (1 - t) * warped_a + t * warped_b
                
                # Noise
                if USE_POISSON_NOISE:
                    img_final = apply_poisson_noise(img_interp)
                else:
                    img_final = img_interp
                
                new_data[current_idx] = img_final
                current_idx += 1
            
            # 2. Originalbild B speichern
            new_data[current_idx] = img_b
            current_idx += 1
            
        print(f"    Serie {idx+1}/{N_series} fertig.", end='\r')

    # Speichern
    group = output_h5.create_group(group_name)
    # Zurück transponieren zu (H, W, N) wie HDF5 es erwartet
    final_data = new_data.transpose(1, 2, 0)
    group.create_dataset("data", data=final_data, compression="gzip")
    print(f"\n -> Gruppe '{group_name}' gespeichert.")

if __name__ == "__main__":
    ROOT_DIR = Path.home()
    IN_DIR  = ROOT_DIR / "data/original_data"
    OUT_DIR = ROOT_DIR / "data/interpolated_data_optical_flow"
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = ["test_data.hdf5"] 
    # Zum Testen erstmal nur eine Datei, später Liste erweitern:
    # ["test_data.hdf5", "training_data.hdf5", "validation_data.hdf5"]

    suffix = "_flow_pois_on" if USE_POISSON_NOISE else "_flow_pois_off"

    for input_name in input_files:
        stem = input_name.replace(".hdf5", "")
        output_name = f"interpolated_{stem}{suffix}.hdf5"
        
        input_path  = IN_DIR / input_name
        output_path = OUT_DIR / output_name
        
        print(f"Processing: {input_name} -> {output_name}")
        
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            for key in ['high_count', 'low_count']:
                process_dataset(key, f_in, f_out)

    print("\nFertig.")