import matplotlib.pyplot as plt
import numpy as np

# Daten
processors = [
    {"name": "Intel 4004", "year": 1971, "count": 2300},
    {"name": "Intel 8086", "year": 1978, "count": 29000},
    {"name": "Intel 486", "year": 1989, "count": 1200000},
    {"name": "Pentium", "year": 1993, "count": 3100000},
    {"name": "Core 2 Duo", "year": 2006, "count": 291000000},
    {"name": "Apple A14", "year": 2020, "count": 11800000000}
]

years = [p["year"] for p in processors]
counts = [p["count"] for p in processors]
names = [p["name"] for p in processors]

# Manuelle Positionierung der Labels (x_offset, y_offset)
offsets = [
    (25, -10),    # Intel 4004
    (-10, 20),   # Intel 8086
    (-15, 20),   # Intel 486
    (25, -15),   # Pentium
    (-20, 20),   # Core 2 Duo
    (-40, 0)    # Apple A14
]

# Plot Setup
fig, ax = plt.subplots(figsize=(14, 9))

# Farben explizit setzen
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Scatter Plot
ax.scatter(years, counts, color='cyan', s=150, zorder=5, label='Processors')

# Moore's Law Trend Line Berechnung
log_counts = np.log10(counts)
coefficients = np.polyfit(years, log_counts, 1)
polynomial = np.poly1d(coefficients)

# --- ÄNDERUNG 1: Zeiträume aufteilen ---
# Historischer Teil (Start bis 2025)
years_historic = np.arange(1971, 2026)
trend_historic = np.power(10, polynomial(years_historic))

# Zukünftiger Teil (2025 bis 2035)
years_future = np.arange(2025, 2036)
trend_future = np.power(10, polynomial(years_future))

# Plot Historisch (Stark)
ax.plot(years_historic, trend_historic, color='orange', linestyle='--', linewidth=3, label="Moore's Law Trend")

# --- ÄNDERUNG 2: Ausdünnende Linie ---
# Plot Zukunft (Dünner, transparenter, gepunktet)
ax.plot(years_future, trend_future, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label="Projection 2035 (?)")

# --- ÄNDERUNG 3: Fragezeichen ---
np.random.seed(42) # Damit es immer gleich aussieht
for yr in years_future[1::2]: # Jedes zweite Jahr im Zukunfts-Array
    # Berechneter Y-Wert auf der Linie
    y_trend = np.power(10, polynomial(yr))
    
    # Zufälliger Versatz (Faktor, da logarithmisch)
    factor = np.random.uniform(0.4, 2.5) 
    # Manchmal invertieren, damit es auch unter der Linie ist
    if np.random.random() > 0.5:
        factor = 1.0 / factor
        
    y_pos = y_trend * factor
    
    ax.text(yr, y_pos, "?", 
            color='orange', 
            fontsize=np.random.randint(14, 24), 
            alpha=0.8,
            fontweight='bold',
            rotation=np.random.randint(-20, 20),
            ha='center', va='center')

# Annotationen
for i, txt in enumerate(names):
    x_offset, y_offset = offsets[i]
    label_text = f"{txt}\n({counts[i]:,})"
    ha = 'right' if x_offset < 0 else 'left'
    
    ax.annotate(label_text, 
                 (years[i], counts[i]),
                 xytext=(x_offset, y_offset), 
                 textcoords='offset points',
                 fontsize=12,
                 fontweight='bold',
                 color='white',
                 ha=ha,
                 arrowprops=dict(arrowstyle="-", color='white', alpha=0.8, linewidth=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7, lw=0.5))

# Achsen-Konfiguration
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')

# --- ÄNDERUNG 4: X-Achse erweitert ---
ax.set_xlim(1965, 2038)

# Titel und Labels
ax.set_title("Moore's Law: Transistor Count to 2035", fontsize=26, fontweight='bold', color='white', pad=30)
ax.set_xlabel("Year", fontsize=18, fontweight='bold', color='white', labelpad=15)
ax.set_ylabel("Transistor Count (Log Scale)", fontsize=18, fontweight='bold', color='white', labelpad=15)

# Ticks Styling
ax.tick_params(axis='both', which='major', labelsize=14, colors='white', length=6, width=2)
ax.tick_params(axis='both', which='minor', colors='gray', length=3)

# Rahmen
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(1.5)

# Legende
legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=14, facecolor='black', edgecolor='white')
for text in legend.get_texts():
    text.set_color("white")

# Layout und Speichern
plt.tight_layout()
plt.savefig("moores_law_2035.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()