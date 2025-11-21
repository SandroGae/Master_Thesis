import PIL.Image
import numpy as np
from PIL import ImageDraw, ImageFont

# === HACK FÜR PILLOW 10.0.0+ ===
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# ===============================

from pathlib import Path
import moviepy.editor as mpy
from moviepy.editor import ColorClip, ImageClip, concatenate_videoclips

# ==========================================
# KONFIGURATION
# ==========================================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS\Plots\Unet\Movies")
SERIES = 50

# Dateinamen
f_irunet  = f"IRUNet_Series_{SERIES}_cut.mp4"
f_3avg    = f"unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006_loss0.0489_val0.0524_averaged_series{SERIES}.mp4"
f_3mid    = f"unet_3d_SSIM_middle_improved_V2__seed42__bf64__D3__lossMAE_SSIM__20251119-144919_loss0.0479_val0.0517_series{SERIES}.mp4"
f_5chan   = f"25D_Model_Series{SERIES}_Depth5.mp4"


# Output Name
out_name = f"combined_series{SERIES}_IRUNET_3stack-avg_3stack-mid_5stack-channel.mp4"
out_path = ROOT_DIR / out_name

# Label-Einstellungen
FONT_SIZE = 30
TXT_COLOR = (255, 255, 255) # Weiß für Pillow
BG_COLOR  = (0, 0, 0)       # Schwarz für Pillow
HEIGHT_TXT = 50             # Höhe des Textbalkens

def create_text_image_clip(text, width, height, duration):
    """
    Erstellt einen Text-Clip mit PIL statt ImageMagick.
    Vermeidet den WinError 2.
    """
    # 1. Bild erstellen (Schwarz)
    img = PIL.Image.new('RGB', (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # 2. Font laden (Versuche Arial, sonst Default)
    try:
        # Pfad zu Arial auf Windows, oder einfach Name versuchen
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
        print("Warnung: Arial nicht gefunden, nutze Standard-Font (klein).")

    # 3. Text zentrieren (Simpel)
    # getbbox gibt (left, top, right, bottom)
    try:
        text_bbox = font.getbbox(text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback für ältere Pillow Versionen
        text_w, text_h = draw.textsize(text, font=font)
        
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    
    # 4. Text zeichnen
    draw.text((x, y), text, font=font, fill=TXT_COLOR)
    
    # 5. In MoviePy Clip wandeln
    # np.array(img) macht daraus ein Array, das MoviePy versteht
    return ImageClip(np.array(img)).set_duration(duration)

def create_labeled_clip(clip, label_text, target_width):
    # 1. Video auf Zielbreite skalieren
    if clip.w != target_width:
        clip = clip.resize(width=target_width)
    
    # 2. Text-Bild erstellen (OHNE ImageMagick)
    txt_clip = create_text_image_clip(label_text, target_width, HEIGHT_TXT, clip.duration)
    
    # 3. Text und Video vertikal stapeln
    return mpy.clips_array([[txt_clip], [clip]])

def main():
    print(f"--- Kombiniere Videos für Serie {SERIES} ---")
    
    # 1. Laden
    print(f"Lade {f_irunet}...")
    c1 = mpy.VideoFileClip(str(ROOT_DIR / f_irunet))
    print(f"Lade {f_3avg}...")
    c2 = mpy.VideoFileClip(str(ROOT_DIR / f_3avg))
    print(f"Lade {f_3mid}...")
    c3 = mpy.VideoFileClip(str(ROOT_DIR / f_3mid))
    print(f"Lade {f_5chan}...")
    c4 = mpy.VideoFileClip(str(ROOT_DIR / f_5chan))

    # Referenzbreite
    ref_w = c1.w
    
    # 2. Sync für das 5-Stack Video (c4)
    print("Resize c4...")
    if c4.w != ref_w:
        c4 = c4.resize(width=ref_w)
    
    frame_duration = 1.0 / c4.fps
    black = ColorClip(size=c4.size, color=(0,0,0), duration=frame_duration)
    
    print("Synce 5-Stack Video...")
    c4_synced = concatenate_videoclips([black, c4, black])

    # 3. Minimale Dauer
    min_dur = min(c.duration for c in [c1, c2, c3, c4_synced])

    # 4. Schneiden & Labeln
    print("Erstelle Labels (via Pillow)...")
    
    final_c1 = create_labeled_clip(c1.subclip(0, min_dur), "IRUNET", ref_w)
    final_c2 = create_labeled_clip(c2.subclip(0, min_dur), "3stack_average", ref_w)
    final_c3 = create_labeled_clip(c3.subclip(0, min_dur), "3stack_middle", ref_w)
    final_c4 = create_labeled_clip(c4_synced.subclip(0, min_dur), "5stack_channel", ref_w)

    # 5. Stapeln
    print("Staple final...")
    final_stack = mpy.clips_array([
        [final_c1],
        [final_c2],
        [final_c3],
        [final_c4]
    ])

    print(f"Schreibe Video: {out_name}")
    final_stack.write_videofile(
        str(out_path),
        fps=c1.fps,
        codec="libx264",
        audio=False,
        threads=4,
        logger="bar"
    )
    print("Fertig.")

if __name__ == "__main__":
    main()