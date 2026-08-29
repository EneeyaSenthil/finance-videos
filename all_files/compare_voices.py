#!/usr/bin/env python3
"""
compare_voices.py -- Generate one short sample per top voice, so you can
listen and compare them back to back instead of running kokoro_tts.py
separately for each one.

USAGE
    python compare_voices.py
    python compare_voices.py --text "Custom sentence to test with"

Produces files like: sample_af_heart.wav, sample_af_bella.wav, etc.
in a folder called voice_samples/
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

# Official grades from Kokoro's own VOICES.md -- best American English voices only
VOICES_TO_TEST = [
    ("af_heart", "A", "female"),
    ("af_bella", "A-", "female"),
    ("af_nicole", "B-", "female"),
    ("am_fenrir", "C+", "male"),
    ("am_michael", "C+", "male"),
    ("am_puck", "C+", "male"),
]

DEFAULT_TEXT = (
    "India's economy grew significantly in recent years, "
    "with manufacturing output surging after new policy reforms."
)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison samples across top Kokoro voices")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Sentence to test with (use your actual script tone)")
    parser.add_argument("--outdir", default="voice_samples", help="Folder to save samples into")
    args = parser.parse_args()

    from kokoro import KPipeline

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("Loading Kokoro model (one-time load, reused for every voice below)...")
    pipeline = KPipeline(lang_code="a")

    for voice, grade, gender in VOICES_TO_TEST:
        print(f"\nGenerating: {voice} (grade {grade}, {gender})...")
        chunks = []
        for gs, ps, audio in pipeline(args.text, voice=voice):
            chunks.append(audio)
        if not chunks:
            print(f"  [skipped -- no audio generated for {voice}]")
            continue
        full_audio = np.concatenate(chunks)
        out_path = outdir / f"sample_{voice}.wav"
        sf.write(out_path, full_audio, 24000)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {len(VOICES_TO_TEST)} samples saved in '{outdir}/'.")
    print("Listen to each and pick your favorite -- then use that voice name")
    print("with kokoro_tts.py for your real narration.")


if __name__ == "__main__":
    main()
