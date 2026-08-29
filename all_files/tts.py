"""
tts.py -- Windows-safe narration generator using edge-tts.

WHY THIS FILE EXISTS
On Windows, running `edge-tts` (or `python -m edge_tts`) directly can crash
with:
    RuntimeError: aiodns needs a SelectorEventLoop on Windows
This happens because a networking library edge-tts depends on defaults to a
mode that doesn't work on Windows. This wrapper fixes it by explicitly
telling Python to use the Windows-compatible networking mode BEFORE edge-tts
loads -- a one-line fix that has to happen first, which the plain command
line can't do for you.

USAGE
    python tts.py --voice en-US-ChristopherNeural --file script.txt --out narration.mp3

LIST AVAILABLE VOICES
    python tts.py --list-voices
"""

import sys
import asyncio

# --- THE ACTUAL FIX: must happen before edge_tts/aiohttp/aiodns load ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import edge_tts


async def list_voices():
    voices = await edge_tts.list_voices()
    american = [v for v in voices if v["Locale"] == "en-US"]
    print(f"Found {len(american)} American English voices:\n")
    for v in american:
        print(f"  {v['ShortName']:30s} {v['Gender']}")


async def synthesize(text, voice, out_path, rate="+0%", volume="+0%", pitch="+0Hz"):
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(out_path)
    print(f"Saved narration to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Windows-safe edge-tts narration generator")
    parser.add_argument("--voice", default="en-US-ChristopherNeural", help="Voice name, e.g. en-US-ChristopherNeural")
    parser.add_argument("--file", help="Path to a .txt file containing your script")
    parser.add_argument("--text", help="Or pass short text directly instead of a file")
    parser.add_argument("--out", default="narration.mp3", help="Output mp3 path")
    parser.add_argument("--list-voices", action="store_true", help="List all American English voices and exit")
    parser.add_argument("--rate", default="+0%", help="Speed adjustment, e.g. +15%% (faster) or -15%% (slower)")
    parser.add_argument("--volume", default="+0%", help="Volume adjustment, e.g. +20%% or -20%%")
    parser.add_argument("--pitch", default="+0Hz", help="Pitch adjustment, e.g. +10Hz or -10Hz")
    args = parser.parse_args()

    if args.list_voices:
        asyncio.run(list_voices())
        return

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        print("Error: provide either --file script.txt or --text \"some words\"")
        sys.exit(1)

    asyncio.run(synthesize(text, args.voice, args.out, args.rate, args.volume, args.pitch))


if __name__ == "__main__":
    main()
