#!/usr/bin/env python3
"""
kokoro_tts.py -- Free, local, human-sounding narration generator.

Runs entirely on your own machine using the open-source Kokoro TTS model.
No API key, no account, no per-use cost, and after the first run (which
downloads the ~habit-forming small model file, a few hundred MB, once)
it needs no internet connection at all.

USAGE
    python kokoro_tts.py --voice af_heart --file script.txt --out narration.wav

LIST VOICES
    python kokoro_tts.py --list-voices

RECOMMENDED VOICES (rated highest for "sounds human" in blind listening tests)
    af_heart    -- female, warm and clear, best all-purpose starting point
    af_bella    -- female, strong balance of realism and stability
    am_michael  -- male, natural American
    am_adam     -- male, natural American, slightly deeper

ONE-TIME SETUP
    pip install kokoro soundfile --break-system-packages
    (first run downloads the model automatically -- needs internet once)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

KNOWN_VOICES = {
    "af_heart": "Female, warm/clear -- best default starting point",
    "af_bella": "Female, strong realism + stability balance",
    "af_nicole": "Female, calm",
    "af_sarah": "Female, neutral American",
    "af_sky": "Female, brighter tone",
    "am_adam": "Male, natural, slightly deeper",
    "am_michael": "Male, natural American",
    "bf_emma": "Female, British accent",
    "bm_george": "Male, British accent",
}


def list_voices():
    print("Known Kokoro voices (lang_code 'a' = American, 'b' = British):\n")
    for name, desc in KNOWN_VOICES.items():
        print(f"  {name:12s} {desc}")
    print("\nStart with af_heart or am_michael, listen, then compare a second one.")


def synthesize(text, voice, out_path, speed=1.0):
    from kokoro import KPipeline

    lang_code = "b" if voice.startswith("b") else "a"
    print(f"Loading Kokoro model (first run downloads it once, then it's local/offline)...")
    pipeline = KPipeline(lang_code=lang_code)

    print(f"Generating speech (voice={voice}, speed={speed})...")
    chunks = []
    for gs, ps, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)

    if not chunks:
        print("Error: no audio was generated -- check your script text isn't empty.")
        sys.exit(1)

    full_audio = np.concatenate(chunks)
    sf.write(out_path, full_audio, 24000)
    print(f"Saved narration to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Free, local, human-sounding narration generator (Kokoro TTS)")
    parser.add_argument("--voice", default="af_heart", help="Voice name, e.g. af_heart, am_michael (see --list-voices). "
                                                              "You can also blend two voices: 'am_michael:60,am_adam:40'")
    parser.add_argument("--file", help="Path to a .txt file containing your script")
    parser.add_argument("--text", help="Or pass short text directly instead of a file")
    parser.add_argument("--out", default="narration.wav", help="Output wav path")
    parser.add_argument("--list-voices", action="store_true", help="List known voices and exit")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed: 1.0 = normal, 1.15 = ~15%% faster/more energetic, "
                              "0.9 = ~10%% slower/more serious. Range roughly 0.5-1.5 stays natural-sounding.")
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Error: provide either --file script.txt or --text \"some words\"")
        sys.exit(1)

    synthesize(text, args.voice, args.out, args.speed)


if __name__ == "__main__":
    main()
