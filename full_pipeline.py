#!/usr/bin/env python3
"""
full_pipeline.py -- The complete chain, one command, script.txt in, video.mp4 out.

    script.txt
        -> Kokoro narration (narration.wav)
        -> volume-boosted narration (loudness normalized)
        -> background music mixed in underneath
        -> Whisper transcribes the mixed narration for ACCURATE word-level
           caption timing (not guessed/evenly split)
        -> one image generated per sentence (Pollinations, free, no key)
        -> Ken Burns pan/zoom motion on each image
        -> word-synced captions layered on top
        -> final video.mp4

USAGE (run this single command on GitHub Codespaces)
    python full_pipeline.py --script script.txt --out video.mp4 --music music.mp3

ONE-TIME SETUP
    pip install kokoro soundfile moviepy numpy pillow faster-whisper

WHY THIS ORDER MATTERS
Whisper transcribes AFTER narration+music are mixed and boosted, not before --
that way the captions are timed against the exact audio that ends up in the
final video, not an earlier draft.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
    VideoClip,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut

BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_bold_font():
    for path in BOLD_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


BOLD_FONT = find_bold_font()


# ---------------------------------------------------------------------------
# STEP 1: Kokoro narration
# ---------------------------------------------------------------------------
def generate_narration(script_text, voice, out_path):
    from kokoro import KPipeline

    print(f"[1/6] Generating narration with Kokoro (voice={voice})...")
    lang_code = "b" if voice.startswith("b") else "a"
    pipeline = KPipeline(lang_code=lang_code)
    chunks = [audio for gs, ps, audio in pipeline(script_text, voice=voice)]
    if not chunks:
        print("Error: Kokoro produced no audio.")
        sys.exit(1)
    full_audio = np.concatenate(chunks)
    sf.write(out_path, full_audio, 24000)
    print(f"      Saved: {out_path}")


# ---------------------------------------------------------------------------
# STEP 2: Boost volume with loudness normalization (ffmpeg)
# ---------------------------------------------------------------------------
def boost_volume(in_path, out_path):
    print("[2/6] Boosting narration volume (loudness normalize)...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path), "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11", str(out_path)],
        check=True, capture_output=True,
    )
    print(f"      Saved: {out_path}")


# ---------------------------------------------------------------------------
# STEP 3: Mix in background music (looped, quiet, no auto-normalize)
# ---------------------------------------------------------------------------
def mix_music(narration_path, music_path, out_path, music_volume):
    print(f"[3/6] Mixing in background music (volume={music_volume})...")
    filter_complex = (
        f"[0:a]volume=1.0[voice];[1:a]volume={music_volume}[music];"
        f"[voice][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(narration_path), "-stream_loop", "-1", "-i", str(music_path),
         "-filter_complex", filter_complex, "-map", "[out]", str(out_path)],
        check=True, capture_output=True,
    )
    print(f"      Saved: {out_path}")


# ---------------------------------------------------------------------------
# STEP 4: Whisper -- accurate word-level timing from the FINAL mixed audio
# ---------------------------------------------------------------------------
def get_word_timings(audio_path):
    print("[4/6] Transcribing final audio for accurate caption sync (Whisper)...")
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    timings = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                timings.append((w.word.strip(), w.start, w.end))
    print(f"      Got {len(timings)} word timestamps")
    return timings


# ---------------------------------------------------------------------------
# STEP 5: One image per sentence (free, no key) + Ken Burns motion
# ---------------------------------------------------------------------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def generate_image_for_sentence(sentence, out_path, width, height):
    """
    Returns the path on success, or None if every real option failed.
    NEVER writes a placeholder -- a placeholder in the final video is
    treated as unacceptable output, not a fallback.
    """
    import time

    prompt = f"professional high quality finance economics b-roll cinematic photo, illustrating: {sentence}, no text, no watermark"

    # --- Option 1: Pollinations.ai (free, no key, rate-paced) ---
    try:
        import urllib.parse
        import urllib.request

        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(out_path, "wb") as f:
            f.write(response.read())
        size = Path(out_path).stat().st_size
        if 10_000 < size < 1_000_000:
            time.sleep(15)
            return out_path
        else:
            print(f"      [Pollinations suspicious size ({size} bytes) -- likely rate-limited, trying next option]")
            Path(out_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"      [Pollinations request failed: {e} -- trying next option]")

    # --- Option 2: Hugging Face Inference API (free, needs HF_API_TOKEN) ---
    hf_token = os.environ.get("HF_API_TOKEN")
    if hf_token:
        try:
            import urllib.request

            hf_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            req = urllib.request.Request(
                hf_url,
                headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"},
                method="POST",
            )
            req.data = f'{{"inputs": {prompt!r}}}'.encode("utf-8")
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            if len(data) > 10_000:
                with open(out_path, "wb") as f:
                    f.write(data)
                return out_path
            else:
                print("      [Hugging Face returned unexpectedly small data]")
        except Exception as e:
            print(f"      [Hugging Face request failed: {e}]")
    else:
        print("      [HF_API_TOKEN not set -- Hugging Face fallback unavailable]")

    # --- Both real options failed. Do NOT create a placeholder. ---
    return None


def ken_burns_clip(image_path, duration, width, height, zoom_in=True, pan="center"):
    """
    Ken Burns effect with both zoom AND a slight directional pan, for a more
    dynamic, engaging feel than a static center zoom alone.
    """
    img = Image.open(image_path).convert("RGB").resize((width, height))
    arr = np.array(img)
    start_scale, end_scale = (1.0, 1.15) if zoom_in else (1.15, 1.0)

    pan_offsets = {
        "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1), "center": (0, 0),
    }
    dx, dy = pan_offsets.get(pan, (0, 0))

    def make_frame(t):
        frac = t / duration if duration > 0 else 0
        scale = start_scale + (end_scale - start_scale) * frac
        h, w = arr.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        frame_img = Image.fromarray(arr).resize((new_w, new_h))
        max_shift_x = (new_w - w) // 2
        max_shift_y = (new_h - h) // 2
        left = (new_w - w) // 2 + int(dx * max_shift_x * frac)
        top = (new_h - h) // 2 + int(dy * max_shift_y * frac)
        left = max(0, min(left, new_w - w))
        top = max(0, min(top, new_h - h))
        return np.array(frame_img.crop((left, top, left + w, top + h)))

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# STEP 6: Captions from real Whisper timings + final assembly
# ---------------------------------------------------------------------------
def build_caption_clips(timings, width, height, accent="#f9c74f"):
    clips = []
    caption_y = int(height * 0.80)
    for word, start, end in timings:
        clean = re.sub(r"[^\w'.,%-]", "", word)
        if not clean:
            continue
        txt = TextClip(
            text=clean.upper(), font=BOLD_FONT, font_size=int(height * 0.06),
            color="white", stroke_color=accent, stroke_width=3, method="label",
        )
        txt = txt.with_start(start).with_duration(max(0.05, end - start))
        txt = txt.with_position(("center", caption_y))
        clips.append(txt)
    return clips


def main():
    p = argparse.ArgumentParser(description="Full pipeline: script.txt -> finished video, one command.")
    p.add_argument("--script", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--music", default=None)
    p.add_argument("--music-volume", type=float, default=0.15)
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--workdir", default="pipeline_tmp")
    args = p.parse_args()

    width, height = (int(x) for x in args.resolution.lower().split("x"))
    workdir = Path(args.workdir)
    workdir.mkdir(exist_ok=True)

    script_text = Path(args.script).read_text(encoding="utf-8")

    # --- Step 1-3: audio chain, RESUMABLE -- skip any stage already done ---
    narration_raw = workdir / "narration_raw.wav"
    narration_boosted = workdir / "narration_boosted.wav"
    final_audio = workdir / "final_audio.mp3"

    if narration_raw.exists():
        print(f"[1/6] Narration already exists, skipping: {narration_raw}")
    else:
        generate_narration(script_text, args.voice, narration_raw)

    if narration_boosted.exists():
        print(f"[2/6] Boosted narration already exists, skipping: {narration_boosted}")
    else:
        boost_volume(narration_raw, narration_boosted)

    if final_audio.exists():
        print(f"[3/6] Final mixed audio already exists, skipping: {final_audio}")
    else:
        if args.music:
            mix_music(narration_boosted, args.music, final_audio, args.music_volume)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", str(narration_boosted), str(final_audio)], check=True, capture_output=True)

    # --- Step 4: whisper timing, RESUMABLE -- cache to a json file ---
    import json
    timings_cache = workdir / "timings.json"
    if timings_cache.exists():
        print(f"[4/6] Caption timings already exist, skipping transcription: {timings_cache}")
        timings = json.loads(timings_cache.read_text())
    else:
        timings = get_word_timings(final_audio)
        timings_cache.write_text(json.dumps(timings))

    # --- Step 5: images, RESUMABLE per-sentence -- and STOPS (no placeholder) on failure ---
    audio_clip = AudioFileClip(str(final_audio))
    duration = audio_clip.duration
    sentences = split_sentences(script_text)
    total_words = sum(len(s.split()) for s in sentences) or 1

    print(f"[5/6] Generating {len(sentences)} images with Ken Burns motion...")
    beat_clips = []
    t_cursor = 0.0
    pans = ["left", "right", "up", "down"]
    for i, sentence in enumerate(sentences):
        weight = len(sentence.split()) / total_words
        beat_duration = max(1.0, duration * weight)
        if t_cursor + beat_duration > duration:
            beat_duration = max(0.5, duration - t_cursor)

        img_path = workdir / f"beat_{i:03d}.png"
        if img_path.exists():
            print(f"      [{i+1}/{len(sentences)}] already generated, skipping: {img_path.name}")
        else:
            print(f"      [{i+1}/{len(sentences)}] generating image for sentence {i+1}...")
            result = generate_image_for_sentence(sentence, img_path, width, height)
            if result is None:
                print(
                    f"\nSTOPPED at sentence {i+1}/{len(sentences)}: both Pollinations and Hugging Face "
                    f"failed to generate a real image for this sentence.\n"
                    f"No placeholder was used -- nothing unusable will end up in your video.\n"
                    f"Everything completed so far (narration, music, captions, and "
                    f"{i} earlier images) is saved in '{args.workdir}/' and will be reused automatically.\n"
                    f"Just run this exact same command again in a little while -- it will pick up\n"
                    f"right here at sentence {i+1} instead of starting over.\n"
                )
                sys.exit(1)

        beat_clips.append(
            ken_burns_clip(img_path, beat_duration, width, height, zoom_in=(i % 2 == 0), pan=pans[i % len(pans)])
        )
        t_cursor += beat_duration
        if t_cursor >= duration:
            break

    # --- Step 6: assemble with crossfade transitions between beats ---
    print("[6/6] Compositing captions and rendering final video...")
    CROSSFADE = 0.5  # seconds of overlap between consecutive images
    faded_clips = []
    for i, clip in enumerate(beat_clips):
        c = clip
        if i > 0:
            c = c.with_effects([CrossFadeIn(CROSSFADE)])
        if i < len(beat_clips) - 1:
            c = c.with_effects([CrossFadeOut(CROSSFADE)])
        faded_clips.append(c)

    background = concatenate_videoclips(faded_clips, method="compose", padding=-CROSSFADE).with_duration(duration)
    caption_clips = build_caption_clips(timings, width, height)
    final = CompositeVideoClip([background] + caption_clips, size=(width, height)).with_duration(duration)
    final = final.with_audio(audio_clip)

    final.write_videofile(args.out, fps=args.fps, codec="libx264", audio_codec="aac", threads=4, preset="medium")
    print(f"\nDone. Final video: {args.out}")


if __name__ == "__main__":
    main()
