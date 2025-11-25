import h5py
import numpy as np
from pathlib import Path
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

def apply_poisson_noise(image_data):
    """
    Simuliert Zählraten-Rauschen (Poisson)
    """
    clean_data = np.maximum(image_data, 0) # Clip auf [0, infinity]
    noisy_data = np.random.poisson(clean_data).astype(np.float32) # Neue Werte ziehen
    return noisy_data

def generate_intermediate_frame_flow(image_a, Image_b):
    """
    Berechnet das Zwischenbild mittels Optical Flow
    L1: Absoluter Fehler pro Pixel --> E = |Intensity(x,y) - Intensity(x+v_x, y+v_y)|
    TV : Regularisierung --> Erlaubt Sprunghafte Bewegungen im Bild
    """
    # Flow berechnen --> attachment=15 wie stark intensitäten variieren dürfen, tight=0.3 hält Kanten scharf, prefilter=False weil es CDW wegglätten könnte
    flow = optical_flow_tvl1(image_a, Image_b, attachment=10, tightness=0.3, num_warp=20, num_iter=50, tol=1e-4, prefilter=False)
    N_rows, N_columns = image_a.shape
    row_coords, col_coords = np.meshgrid(np.arange(N_rows), np.arange(N_columns), indexing='ij') # Warp Grid erstellen
    
    # Warping
    coordinates_a = np.array([row_coords - flow[0]*0.5, col_coords - flow[1]*0.5]) # Von wo kam der Pixel? -> minus halber Flow Vektor
    coordinates_b = np.array([row_coords + flow[0]*0.5, col_coords + flow[1]*0.5]) # Wohin geht der Pixel? -> plus halber Flow Vektor
    image_a_warped = warp(image_a, coordinates_a, mode='edge') # Bilder verziehen (warpen)
    Image_b_warped = warp(Image_b, coordinates_b, mode='edge')
    mixed_image = 0.5 * image_a_warped + 0.5 * Image_b_warped # Mischen
 
    return mixed_image

def process_dataset(group_name, input_h5, output_h5):
    """
    Unterteilt Datenset in 41er-Gruppen und interpoliert ein Bild zwischen zwei bestehenden Bildern durch Optical Flow
    """
    dataset = input_h5[f"{group_name}/data"][:]

    Height, Width, N_total = dataset.shape
    series_length = 41
    N_series = N_total // series_length
    data_transposed = dataset.transpose(2, 0, 1) # (H, W, N) --> (N, H, W)
    
    # Array erstellen und befüllen
    new_series_len = series_length + (series_length - 1) # 41+40=81
    N_total_new = N_series * new_series_len
    new_data = np.zeros((N_total_new, Height, Width), dtype=np.float32)

    current_idx = 0
    for idx in range(0, N_series, 1):
        start = idx * series_length
        end = start + series_length
        image_series = data_transposed[start:end]

        new_data[current_idx] = image_series[0] # Startbild
        current_idx += 1
        print(f" -> Fortschritt: Serie {idx+1} von {N_series} fertig.", flush=True)
        
        for i in range(series_length - 1):
            image_a = image_series[i]
            image_b = image_series[i+1]
            
            image_interp_clean = generate_intermediate_frame_flow(image_a, image_b) # Berechne das saubere, verschobene Bild
            image_interp_noisy = apply_poisson_noise(image_interp_clean) # Noise wiederherstellen
            
            # Speichern
            new_data[current_idx] = image_interp_noisy # Interpoliertes Bild
            current_idx += 1
            new_data[current_idx] = image_b # Original Bild
            current_idx += 1

    group = output_h5.create_group(group_name)
    final_data = new_data.transpose(1, 2, 0) # Zurück zu (H, W, N)
    group.create_dataset("data", data=final_data, compression="gzip")
    print(f" -> Gruppe '{group_name}' erfolgreich gespeichert.")


if __name__ == "__main__":
    # Pfade definieren
    ROOT_DIR = Path.home()
    IN_DIR  = ROOT_DIR / "data/original_data"
    OUT_DIR = ROOT_DIR / "data/interpolated_data_optical_flow"

    datasets = [("test_data.hdf5",        "interpolated_test_data.hdf5"),
                ("training_data.hdf5",    "interpolated_training_data.hdf5"),
                ("validation_data.hdf5",  "interpolated_validation_data.hdf5")]

    # Loop über alle drei Dateien
    for input_name, output_name in datasets:
        input_path  = IN_DIR / input_name
        output_path = OUT_DIR / output_name

        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            for key in ['high_count', 'low_count']:
                process_dataset(key, f_in, f_out)
    
    print("\nAlle Jobs erledigt.")