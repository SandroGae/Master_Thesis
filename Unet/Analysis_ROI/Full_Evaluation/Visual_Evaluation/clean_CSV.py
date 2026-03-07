import pandas as pd
from pathlib import Path

# =====================================================
# 1. KONFIGURATION
# =====================================================
# Definiere die Pfade (analog zu deinem Setup)
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "Full_Evaluation_Results_Extended.csv"

# Die "Dirty Four" Serien-IDs, die gekickt werden sollen
KICK_SERIES = [13, 15, 67, 73]

# =====================================================
# 2. BEREINIGUNG
# =====================================================
if not CSV_FILE.exists():
    print(f"Fehler: Datei nicht gefunden unter {CSV_FILE}")
else:
    print(f"Lese Datei ein: {CSV_FILE.name}")
    df = pd.read_csv(CSV_FILE)
    
    initial_rows = len(df)
    
    # Filtere alle Zeilen heraus, deren SeriesID in der Kick-Liste ist
    # ~ (Tilde) bedeutet "NICHT"
    df_cleaned = df[~df['SeriesID'].isin(KICK_SERIES)]
    
    final_rows = len(df_cleaned)
    removed_rows = initial_rows - final_rows
    
    # =====================================================
    # 3. SPEICHERN
    # =====================================================
    # Wir speichern es unter dem gleichen Namen (überschreiben)
    df_cleaned.to_csv(CSV_FILE, index=False)
    
    print("-" * 50)
    print(f"Erfolg!")
    print(f"Ursprüngliche Zeilen: {initial_rows}")
    print(f"Entfernte Zeilen:      {removed_rows} (Serien: {KICK_SERIES})")
    print(f"Verbleibende Zeilen:   {final_rows}")
    print(f"Datei wurde aktualisiert: {CSV_FILE}")
    print("-" * 50)