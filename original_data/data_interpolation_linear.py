import h5py
import numpy as np
from pathlib import Path


USE_POISSON_NOISE = False


def apply_poisson_noise(image_data):
    """
    Simuliert Zählraten-Rauschen (Poisson)
    """
    clean_data = np.maximum(image_data, 0) # Clip auf [0, infinity]
    noisy_data = np.random.poisson(clean_data).astype(np.float32) # Neue Werte ziehen
    return noisy_data

def process_dataset(group_name, input_h5, output_h5, use_noise=True):
    """
    Unterteilt Datenset in 41er-Gruppen und interpoliert ein Bild 
    zwischen zwei bestehenden Bildern durch lineare Interpolation.
    use_noise: Wenn True, wird Poisson Noise auf die interpolierten Bilder angewendet.
    """
    N_interpolate = 5 # Anzahl Zwischenbilder

    dataset = input_h5[f"{group_name}/data"][:]

    Height, Width, N_total = dataset.shape
    series_length = 41
    N_series = N_total // series_length
    data_transposed = dataset.transpose(2, 0, 1) # (H, W, N) --> (N, H, W)
    
    # Array erstellen und befüllen
    # Berechnung der neuen Länge basierend auf N_interpolate
    new_series_len = series_length + (series_length - 1) * N_interpolate 
    N_total_new = N_series * new_series_len
    new_data = np.zeros((N_total_new,  Height, Width), dtype=np.float32)

    current_idx = 0
    for idx in range(0, N_series, 1):
        start = idx * series_length
        end = start + series_length
        image_series = data_transposed[start:end]

        new_data[current_idx] = image_series[0] # Startbild
        current_idx += 1
        
        for i in range(series_length - 1):
            image_a = image_series[i]
            image_b = image_series[i+1]
            
            # Loop über die N Zwischenbilder
            for j in range(1, N_interpolate + 1):
                # Berechne den Anteil von Bild B (linear steigend von 0 bis 1)
                alpha = j / (N_interpolate + 1)

                image_interp = (1 - alpha) * image_a + alpha * image_b    # Interpolation clean
                
                # Toggle Check: Noise nur anwenden, wenn gewünscht
                if use_noise:
                    image_to_save = apply_poisson_noise(image_interp)
                else:
                    image_to_save = image_interp
                
                new_data[current_idx] = image_to_save  # Interpoliertes Bild
                current_idx += 1

            new_data[current_idx] = image_b             # Original Bild
            current_idx += 1

    group = output_h5.create_group(group_name)
    final_data = new_data.transpose(1, 2, 0)
    group.create_dataset("data", data=final_data, compression="gzip")
    print(f"    -> Gruppe '{group_name}' erfolgreich gespeichert (Noise={use_noise}).")


if __name__ == "__main__":
    # Pfade definieren
    ROOT_DIR = Path.home()
    IN_DIR  = ROOT_DIR / "data/original_data"
    OUT_DIR = ROOT_DIR / "data/interpolated_data_linear"
    
    # Sicherstellen, dass Output-Ordner existiert
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Suffix basierend auf Toggle bestimmen
    suffix = "_pois_on" if USE_POISSON_NOISE else "_pois_off"
    print(f"Starte Verarbeitung. Modus: {suffix.strip('_').upper()}")

    # Liste der Eingabedateien
    input_files = ["test_data.hdf5", 
                   "training_data.hdf5", 
                   "validation_data.hdf5"]

    # Loop über alle Dateien
    for input_name in input_files:
        # Output Name dynamisch zusammenbauen
        stem = input_name.replace(".hdf5", "")
        output_name = f"interpolated_{stem}{suffix}.hdf5"

        input_path  = IN_DIR / input_name
        output_path = OUT_DIR / output_name
        
        print(f"Verarbeite: {input_name} -> {output_name}")

        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            for key in ['high_count', 'low_count']:
                process_dataset(key, f_in, f_out, use_noise=USE_POISSON_NOISE)
    
    print("\nAlle Jobs erledigt.")