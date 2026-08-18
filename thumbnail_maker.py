#!/usr/bin/env python3
"""
thumbnail_maker.py -- Generates 2-3 candidate YouTube thumbnails from a
video's own generated images, with bold branded text overlay.

WHY THIS APPROACH
A high-CTR thumbnail needs: a strong, high-contrast focal image, bold
minimal text (3-5 words), and consistent channel branding. Rather than
generating a brand-new image blind, this reuses your video's own real
generated images (from pipeline_tmp/beat_*.png) -- so the thumbnail
actually represents what's in the video, not a disconnected graphic.

USAGE
    python thumbnail_maker.py --workdir pipeline_tmp --text "SHOULD YOU EAT SALAD?"

Produces 3 candidates: thumbnail_1.png, thumbnail_2.png, thumbnail_3.png
in the current directory -- pick whichever looks strongest.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Brand colors -- matches the channel's navy/red/white identity
NAVY = (13, 27, 62)
RED = (196, 30, 42)
WHITE = (255, 255, 255)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_bold_font():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


BOLD_FONT = find_bold_font()

THUMBNAIL_SIZE = (1280, 720)  # YouTube's standard thumbnail size


def pick_source_image(workdir):
    """
    Picks a real generated image from the video to use as the thumbnail base.
    Prefers a middle-of-video image (often more visually interesting than
    the opening beat) but falls back to whatever's available.
    """
    beats = sorted(Path(workdir).glob("beat_*.png"))
    if not beats:
        raise FileNotFoundError(
            f"No beat_*.png images found in {workdir} -- run the main pipeline first "
            f"so there are real generated images to build a thumbnail from."
        )
    # Prefer something from the middle third of the video
    mid_start = len(beats) // 3
    mid_end = max(mid_start + 1, (len(beats) * 2) // 3)
    candidates = beats[mid_start:mid_end] or beats
    return random.choice(candidates)


def darken_for_text(img, amount=0.45):
    """Darkens the image so white bold text stays readable on top of it."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1 - amount)


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_thumbnail(source_path, text, out_path, style="bottom_banner"):
    img = Image.open(source_path).convert("RGB")
    img = img.resize(THUMBNAIL_SIZE)

    if style == "bottom_banner":
        img = darken_for_text(img, amount=0.15)  # only darken slightly, banner handles contrast
        draw = ImageDraw.Draw(img, "RGBA")
        banner_height = int(THUMBNAIL_SIZE[1] * 0.32)
        banner_top = THUMBNAIL_SIZE[1] - banner_height
        draw.rectangle([0, banner_top, THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1]], fill=(*NAVY, 235))
        draw.rectangle([0, banner_top, THUMBNAIL_SIZE[0], banner_top + 8], fill=RED)

        font_size = int(banner_height * 0.32)
        font = ImageFont.truetype(BOLD_FONT, font_size) if BOLD_FONT else ImageFont.load_default()
        lines = wrap_text(text.upper(), font, THUMBNAIL_SIZE[0] - 80, draw)
        total_text_height = len(lines) * (font_size + 10)
        y = banner_top + (banner_height - total_text_height) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (THUMBNAIL_SIZE[0] - w) // 2
            draw.text((x, y), line, font=font, fill=WHITE)
            y += font_size + 10

    elif style == "top_banner":
        img = darken_for_text(img, amount=0.15)
        draw = ImageDraw.Draw(img, "RGBA")
        banner_height = int(THUMBNAIL_SIZE[1] * 0.30)
        draw.rectangle([0, 0, THUMBNAIL_SIZE[0], banner_height], fill=(*NAVY, 235))
        draw.rectangle([0, banner_height - 8, THUMBNAIL_SIZE[0], banner_height], fill=RED)

        font_size = int(banner_height * 0.34)
        font = ImageFont.truetype(BOLD_FONT, font_size) if BOLD_FONT else ImageFont.load_default()
        lines = wrap_text(text.upper(), font, THUMBNAIL_SIZE[0] - 80, draw)
        total_text_height = len(lines) * (font_size + 10)
        y = (banner_height - total_text_height) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (THUMBNAIL_SIZE[0] - w) // 2
            draw.text((x, y), line, font=font, fill=WHITE)
            y += font_size + 10

    else:  # "stroke_overlay" -- bold stroked text directly over the darkened image, no banner block
        img = darken_for_text(img, amount=0.35)
        draw = ImageDraw.Draw(img)
        font_size = int(THUMBNAIL_SIZE[1] * 0.14)
        font = ImageFont.truetype(BOLD_FONT, font_size) if BOLD_FONT else ImageFont.load_default()
        lines = wrap_text(text.upper(), font, THUMBNAIL_SIZE[0] - 100, draw)
        total_text_height = len(lines) * (font_size + 15)
        y = (THUMBNAIL_SIZE[1] - total_text_height) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (THUMBNAIL_SIZE[0] - w) // 2
            draw.text((x, y), line, font=font, fill=WHITE, stroke_width=6, stroke_fill=RED)
            y += font_size + 15

    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate candidate YouTube thumbnails from your video's real images")
    parser.add_argument("--workdir", default="pipeline_tmp", help="Folder containing beat_*.png images from the main pipeline")
    parser.add_argument("--text", required=True, help="Thumbnail text, e.g. 'SHOULD YOU EAT SALAD?'")
    parser.add_argument("--out-prefix", default="thumbnail")
    args = parser.parse_args()

    styles = ["bottom_banner", "top_banner", "stroke_overlay"]
    print(f"Generating {len(styles)} thumbnail candidates from real images in {args.workdir}/...")

    for i, style in enumerate(styles, 1):
        try:
            source = pick_source_image(args.workdir)
            out_path = f"{args.out_prefix}_{i}.png"
            make_thumbnail(source, args.text, out_path, style=style)
            print(f"  {i}. {out_path}  (style: {style}, based on {source.name})")
        except Exception as e:
            print(f"  [{i}. {style} failed: {e}]")

    print("\nDone. Open each in an image viewer/Notepad-adjacent tool and pick the strongest one.")


if __name__ == "__main__":
    main()
