import PIL.Image
import numpy as np
from PIL import ImageDraw, ImageFont
from pathlib import Path
import moviepy.editor as mpy

# Hack für Pillow 10+
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# ================= CONFIG =================
ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS\Plots\Unet\Movies")
SERIES = 12
OUT_NAME = f"combined_series{SERIES}_dynamic_stack.mp4"

# Format: "Label im Video": "Dateiname"
INPUT_FILES = {
    "IRUNET":           f"IRUNet_Series_12_cut.mp4",
    "5stack_middle_V2":   f"25D_Model_Series12_Depth5.mp4",
    # "5stack_middle_V2_interpolated":    f"25D_Model_Series12_Depth5_interpolated.mp4",
    # "5stack_middle_V2_kernel_3x5":   f"25D_Model_Series12_Depth5_kernel_3x5.mp4"
}

FONT_OPTS = {"size": 30, "color": (255, 255, 255), "bg": (0, 0, 0), "height": 50}

# ================= LOGIC =================
def create_label_clip(text, width, duration):
    img = PIL.Image.new('RGB', (width, FONT_OPTS["height"]), color=FONT_OPTS["bg"])
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", FONT_OPTS["size"])
    except OSError:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    txt_w, txt_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - txt_w) // 2, (FONT_OPTS["height"] - txt_h) // 2), text, font=font, fill=FONT_OPTS["color"])
    
    return mpy.ImageClip(np.array(img)).set_duration(duration)

def main():
    print(f"--- Verarbeite {len(INPUT_FILES)} Videos ---")
    
    loaded_clips = []
    
    # 1. Videos laden
    for label, fname in INPUT_FILES.items():
        path = ROOT_DIR / fname
        if path.exists():
            print(f"Lade: {label}")
            loaded_clips.append({"clip": mpy.VideoFileClip(str(path)), "label": label})
        else:
            print(f"WARNUNG: Datei nicht gefunden: {fname}")

    if not loaded_clips: return

    # 2. Parameter bestimmen (Ref-Breite & Min-Dauer)
    ref_w = loaded_clips[0]["clip"].w
    min_dur = min(c["clip"].duration for c in loaded_clips)
    
    final_units = []

    # 3. Verarbeiten & Labeln
    for item in loaded_clips:
        vid = item["clip"]
        
        # Resize & Cut
        if vid.w != ref_w: vid = vid.resize(width=ref_w)
        vid = vid.subclip(0, min_dur)
        
        # Label erstellen & stapeln
        lbl = create_label_clip(item["label"], ref_w, min_dur)
        unit = mpy.clips_array([[lbl], [vid]]) # Vertikal: Label über Video
        final_units.append(unit)

    # 4. Alles vertikal stapeln
    print(f"Staple Videos und speichere nach {OUT_NAME}...")
    final_stack = mpy.clips_array([[u] for u in final_units]) # [[u1], [u2], ...] für vertikalen Stack

    final_stack.write_videofile(
        str(ROOT_DIR / OUT_NAME),
        fps=loaded_clips[0]["clip"].fps,
        codec="libx264",
        audio=False,
        threads=4,
        logger="bar"
    )

if __name__ == "__main__":
    main()