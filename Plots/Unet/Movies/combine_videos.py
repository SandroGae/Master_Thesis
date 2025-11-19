from pathlib import Path
import moviepy.editor as mpy

ROOT_DIR = Path(r"C:\Users\sandr\VS_MASTER_THESIS\Plots\Unet\Movies")

SERIES = 50

top_path = ROOT_DIR / f"unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006_loss0.0489_val0.0524_series{SERIES}.mp4"
mid_path = ROOT_DIR / f"unet_3d_SSIM__seed42__bf64__D3__lossMAE_SSIM__20251112-113006_loss0.0489_val0.0524_averaged_series{SERIES}.mp4"
bot_path = ROOT_DIR / f"unet_3d_SSIM_middle__seed42__bf64__D3__lossMAE_SSIM__20251112-231304_loss0.0481_val0.0518_series{SERIES}.mp4"
out_path = ROOT_DIR / f"combined_series{SERIES}_top0524_mid0524avg_bot0518.mp4"


def main():
    top = mpy.VideoFileClip(str(top_path))
    mid = mpy.VideoFileClip(str(mid_path))
    bot = mpy.VideoFileClip(str(bot_path))

    # Alle auf gleiche Dauer schneiden
    min_duration = min(c.duration for c in (top, mid, bot))
    top = top.subclip(0, min_duration)
    mid = mid.subclip(0, min_duration)
    bot = bot.subclip(0, min_duration)

    # KEIN resize – Auflösung ist gleich
    stacked = mpy.clips_array([[top], [mid], [bot]])

    stacked.write_videofile(
        str(out_path),
        fps=top.fps,
        codec="libx264",
        audio=False,
        threads=4,
    )


if __name__ == "__main__":
    main()
