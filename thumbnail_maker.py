#!/usr/bin/env python3
"""
thumbnail_maker.py -- Generates a purpose-built, high-CTR YouTube thumbnail,
using the SAME LLM that already understands your whole script and style
guide from the main pipeline -- not a random image grabbed from the video.

WHY THIS APPROACH
A high-CTR thumbnail needs a deliberately chosen "money shot" -- the single
most curiosity-driving visual moment in the story -- not whichever beat
image happened to land in the middle third of the video. Since the LLM
already read your entire script to build the style guide, it's in a much
better position to identify that moment than a random pick. Uses the same
Gemini-based text and image generation as the main pipeline (GEMINI_API_KEY).

ON TEXT/TITLE CONTROL -- IMPORTANT
The LLM SUGGESTS 3 short candidate titles. It never picks for you. You
always choose or rewrite the final text yourself via --text.

TWO-STEP WORKFLOW
  Step 1: get title suggestions (no image generated yet, fast) -- just
  leave out --text and it stops after showing suggestions:
      python thumbnail_maker.py --script script.txt --workdir pipeline_tmp

  Step 2: once you've picked/written your title, generate the real thumbnails
      python thumbnail_maker.py --script script.txt --workdir pipeline_tmp --text "YOUR CHOSEN TITLE"

Produces 3 candidates: thumbnail_1.png, thumbnail_2.png, thumbnail_3.png --
pick whichever looks strongest.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# Reuse the already-tested LLM call and image-generation chain from the main
# pipeline instead of duplicating that logic here.
from full_pipeline import call_llm, generate_image_for_sentence

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


def generate_thumbnail_concept(script_text, style_guide, workdir):
    """
    Asks the LLM (which already understands the whole script + style guide)
    to identify the single most curiosity-driving visual moment for a
    thumbnail, write a dedicated image prompt for it, and suggest 3 short
    candidate titles. Cached to disk (resumable).

    Returns (image_prompt, title_suggestions list).
    """
    cache_path = workdir / "thumbnail_concept.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return data["image_prompt"], data["title_suggestions"]

    print("Analyzing your script for the strongest thumbnail concept...")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print(
            "\nSTOPPED: GEMINI_API_KEY is not set, so a thumbnail concept can't be generated.\n"
            "Set it with: export GEMINI_API_KEY=\"your-key-here\"\n"
        )
        sys.exit(1)

    prompt = (
        "You are a YouTube thumbnail strategist for an American finance/economics "
        "documentary channel. High-CTR thumbnails need ONE deliberately chosen, "
        "curiosity-driving visual moment -- not a random scene.\n\n"
        f"STYLE GUIDE for this video:\n{style_guide}\n\n"
        f"FULL SCRIPT:\n{script_text[:3000]}\n\n"
        "1. Identify the single most visually striking, curiosity-driving moment "
        "in this story -- the one image that would make someone stop scrolling.\n"
        "2. Write ONE detailed, concrete, photorealistic image-generation prompt "
        "for that specific moment (not a person posing, not generic stock imagery -- "
        "a real, specific scene tied directly to the story). Follow the style guide.\n"
        "3. Suggest 3 short candidate thumbnail titles (3-6 words each, ALL CAPS, "
        "punchy, curiosity-driven -- these are SUGGESTIONS ONLY, the creator will "
        "choose or rewrite).\n\n"
        "Output ONLY a JSON object with exactly this shape, nothing else:\n"
        '{"image_prompt": "...", "title_suggestions": ["...", "...", "..."]}'
    )

    try:
        import re
        raw = call_llm(prompt, gemini_key, max_tokens=400, temperature=0.8, min_content_length=40)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(cleaned)

        image_prompt = data["image_prompt"]
        title_suggestions = data["title_suggestions"]
        if not image_prompt or not isinstance(title_suggestions, list) or len(title_suggestions) == 0:
            raise ValueError(f"unexpected response shape: {data!r}")

        cache_path.write_text(json.dumps({
            "image_prompt": image_prompt,
            "title_suggestions": title_suggestions,
        }))
        return image_prompt, title_suggestions

    except Exception as e:
        print(
            f"\nSTOPPED: thumbnail concept generation failed with an error: {e}\n"
            f"Run this exact same command again once it's fixed.\n"
        )
        sys.exit(1)


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
    parser = argparse.ArgumentParser(description="Generate a purpose-built, high-CTR YouTube thumbnail")
    parser.add_argument("--script", required=True, help="Path to the video's script.txt")
    parser.add_argument("--workdir", default="pipeline_tmp", help="Folder with the cached style_guide.txt from the main pipeline")
    parser.add_argument("--text", help="Your chosen/rewritten thumbnail title. If omitted, shows suggestions and stops.")
    parser.add_argument("--out-prefix", default="thumbnail")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    style_guide_path = workdir / "style_guide.txt"
    if not style_guide_path.exists():
        print(
            f"Error: {style_guide_path} not found -- run full_pipeline.py on this script "
            f"first, so a style guide exists for the thumbnail concept to build on."
        )
        sys.exit(1)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print(
            "\nSTOPPED: GEMINI_API_KEY is not set.\n"
            "Set it with: export GEMINI_API_KEY=\"your-key-here\"\n"
        )
        sys.exit(1)

    script_text = Path(args.script).read_text(encoding="utf-8")
    style_guide = style_guide_path.read_text(encoding="utf-8")

    image_prompt, title_suggestions = generate_thumbnail_concept(script_text, style_guide, workdir)

    if not args.text:
        print("\nThumbnail concept ready. Suggested titles (pick one, edit one, or write your own):\n")
        for i, t in enumerate(title_suggestions, 1):
            print(f"  {i}. {t}")
        print(
            f"\nRun this again with --text \"YOUR CHOSEN TITLE\" to generate the actual "
            f"thumbnail images.\nExample:\n"
            f"  python thumbnail_maker.py --script {args.script} --workdir {args.workdir} "
            f"--text \"{title_suggestions[0]}\"\n"
        )
        return

    concept_image_path = workdir / "thumbnail_concept.png"
    if concept_image_path.exists():
        print(f"Concept image already generated, skipping: {concept_image_path}")
    else:
        print("Generating the dedicated thumbnail concept image...")
        result = generate_image_for_sentence(image_prompt, concept_image_path, THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1])
        if result is None:
            print(
                "\nSTOPPED: couldn't generate the thumbnail concept image -- Pollinations "
                "failed after multiple attempts. No placeholder was used. Run this exact "
                "same command again in a little while.\n"
            )
            sys.exit(1)

    styles = ["bottom_banner", "top_banner", "stroke_overlay"]
    print(f"\nGenerating {len(styles)} text-treatment variants of your title on the concept image...")
    for i, style in enumerate(styles, 1):
        out_path = f"{args.out_prefix}_{i}.png"
        make_thumbnail(concept_image_path, args.text, out_path, style=style)
        print(f"  {i}. {out_path}  (style: {style})")

    print("\nDone. Open each and pick the strongest one.")


if __name__ == "__main__":
    main()
