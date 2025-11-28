import pydicom
import os
import shutil
from collections import defaultdict

# ---------------------------------------------------------
# PFAD ANPASSEN: Wo liegen deine 18.000 Bilder?
source_folder = "original_data/1.000000-NA-55980"
# ---------------------------------------------------------

# Hier speichern wir, welche mA-Werte wir finden
dose_groups = defaultdict(list)

print(f"Scanne {source_folder} nach Röntgen-Dosis (mA)...")

files = [f for f in os.listdir(source_folder) if f.endswith('.dcm')]
total_files = len(files)

for i, filename in enumerate(files):
    filepath = os.path.join(source_folder, filename)
    try:
        # Header lesen
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        
        # Wir holen uns die Stromstärke (mA)
        # Tag (0018,1151) XRayTubeCurrent
        if 'XRayTubeCurrent' in ds:
            ma_value = int(ds.XRayTubeCurrent) # Als ganze Zahl (z.B. 319)
            dose_groups[ma_value].append(filename)
        else:
            # Falls kein mA Tag da ist, versuchen wir Exposure (mAs)
            if 'Exposure' in ds:
                ma_value = int(ds.Exposure)
                dose_groups[f"mAs_{ma_value}"].append(filename)
            else:
                dose_groups["Unknown_Dose"].append(filename)

    except Exception as e:
        print(f"Fehler bei {filename}")

    if i % 2000 == 0:
        print(f"{i}/{total_files} gescannt...")

print("-" * 40)
print("ERGEBNIS DER ANALYSE:")
print("-" * 40)

# Wir zeigen an, was wir gefunden haben
for dose, file_list in dose_groups.items():
    print(f"• Dosis-Wert: {dose} mA (oder mAs) -> Anzahl Bilder: {len(file_list)}")

print("-" * 40)
print("INTERPRETATION:")
sorted_keys = sorted([k for k in dose_groups.keys() if isinstance(k, int)], reverse=True)

if len(sorted_keys) >= 2:
    print(f"HÖCHSTER WERT ({sorted_keys[0]} mA) = Wahrscheinlich GROUND TRUTH (Full Dose)")
    print(f"NIEDRIGE WERTE ({sorted_keys[1:]} mA) = Wahrscheinlich INPUT (Low Dose/Noisy)")
    
    # Frage user ob wir sortieren sollen
    print("\nSoll ich die Dateien jetzt in Ordner 'GT' und 'Input' sortieren? (Das Skript beendet sich hier erst mal zur Sicherheit)")
else:
    print("Konnte keine zwei verschiedenen Dosis-Gruppen finden. Bitte Output prüfen.")