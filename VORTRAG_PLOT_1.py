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

# Plot Setup - explizit mit Figure und Axes Objekt arbeiten für volle Kontrolle
fig, ax = plt.subplots(figsize=(14, 9))

# Farben explizit setzen (Sicherheitshalber, falls Dark Mode nicht greift)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Scatter Plot
ax.scatter(years, counts, color='cyan', s=150, zorder=5, label='Processors')

# Moore's Law Trend Line
log_counts = np.log10(counts)
coefficients = np.polyfit(years, log_counts, 1)
polynomial = np.poly1d(coefficients)
trend_y = np.power(10, polynomial(years))

ax.plot(years, trend_y, color='orange', linestyle='--', linewidth=3, label="Moore's Law Trend")

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

# Titel und Labels - explizit in Weiß
ax.set_title("Moore's Law: Transistor Count Over Time", fontsize=26, fontweight='bold', color='white', pad=30)
ax.set_xlabel("Year", fontsize=18, fontweight='bold', color='white', labelpad=15)
ax.set_ylabel("Transistor Count (Log Scale)", fontsize=18, fontweight='bold', color='white', labelpad=15)

# Ticks (Zahlen an den Achsen) explizit stylen
ax.tick_params(axis='both', which='major', labelsize=14, colors='white', length=6, width=2)
ax.tick_params(axis='both', which='minor', colors='gray', length=3)

# Rahmen (Spines) einfärben
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(1.5)

# Legende
legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=14, facecolor='black', edgecolor='white')
for text in legend.get_texts():
    text.set_color("white")

# Layout und Speichern
plt.tight_layout()

# Wichtig: facecolor beim Speichern mitgeben
plt.savefig("moores_law_final.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()