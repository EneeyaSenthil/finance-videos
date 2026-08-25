#!/usr/bin/env python3
"""
full_pipeline.py -- The complete chain, one command, script.txt in, video.mp4 out.

    script.txt
        -> Kokoro narration (narration.wav)
        -> volume-boosted narration (loudness normalized)
        -> background music mixed in underneath
        -> Whisper transcribes the mixed narration for ACCURATE word-level
           caption timing (not guessed/evenly split)
        -> full script read + visual style guide generated (Gemini, free tier)
        -> one detailed, camera-aware image prompt written per sentence (Gemini)
        -> one image generated per sentence (Gemini image model, free tier)
        -> Ken Burns pan/zoom motion on each image
        -> word-synced captions layered on top
        -> final video.mp4

USAGE (run this single command on GitHub Codespaces)
    python full_pipeline.py --script script.txt --out video.mp4 --music music.mp3

ONE-TIME SETUP
    pip install kokoro soundfile moviepy numpy pillow faster-whisper
    export GEMINI_API_KEY="your-key-from-aistudio.google.com/apikey"

WHY THIS ORDER MATTERS
Whisper transcribes AFTER narration+music are mixed and boosted, not before --
that way the captions are timed against the exact audio that ends up in the
final video, not an earlier draft.
"""

import argparse
import json
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
def chunk_script(script_text, max_chars=4000):
    """
    Splits the script into sentence-boundary-safe chunks of roughly max_chars
    each, so a script of ANY length (5,000 / 10,000 / 20,000+ characters) can
    be read in full without truncation. Each chunk is a clean run of whole
    sentences -- never a mid-sentence cut.
    """
    sentences = split_sentences(script_text)
    chunks = []
    current, current_len = [], 0
    for s in sentences:
        if current and current_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(s)
        current_len += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_script_chunk(chunk_text, chunk_index, total_chunks, api_key):
    """
    One piece of the map-reduce read: extract plain notes from a single slice
    of the script -- concrete subject matter, named entities/numbers, tone,
    and any strong visual imagery implied by the words. No topic-specific
    wording baked in here; this must work for ANY script, on any subject.
    """
    prompt = (
        f"This is part {chunk_index + 1} of {total_chunks} of a longer video script "
        "for a YouTube channel. Read this part carefully and write plain notes (under "
        "80 words) covering: the specific subject matter discussed in this part, any "
        "concrete people/places/companies/numbers/events named, the emotional tone, and "
        "any strong visual imagery implied by the words. Output ONLY the notes, no "
        "preamble, no restating these instructions.\n\n"
        f"SCRIPT PART:\n{chunk_text}"
    )
    return call_llm(prompt, api_key, max_tokens=150, temperature=0.4, min_content_length=15)


def read_full_script(script_text, workdir, api_key):
    """
    Reads the ENTIRE script, regardless of length, via a resumable map-reduce
    pass: slice into chunks -> summarize each chunk -> return the combined
    notes covering the full runtime of the video, not just the opening.
    """
    notes_cache = workdir / "script_notes.json"
    chunks = chunk_script(script_text, max_chars=4000)

    if len(chunks) == 1:
        # Short script: no need to summarize, the whole thing fits in one call.
        return script_text

    notes = []
    if notes_cache.exists():
        notes = json.loads(notes_cache.read_text())

    if len(notes) >= len(chunks):
        print(f"      Full-script read already done ({len(chunks)} parts), reusing.")
        return "\n".join(f"Part {i + 1}: {n}" for i, n in enumerate(notes[:len(chunks)]))

    print(f"      Script is long -- reading it in {len(chunks)} parts so nothing gets skipped...")
    for i in range(len(notes), len(chunks)):
        print(f"      Reading part {i + 1}/{len(chunks)}...")
        try:
            note = summarize_script_chunk(chunks[i], i, len(chunks), api_key)
        except Exception as e:
            print(f"\nSTOPPED: reading part {i + 1}/{len(chunks)} of the script failed: {e}\n"
                  f"Run this exact same command again -- parts already read are saved.\n")
            sys.exit(1)
        notes.append(note)
        notes_cache.write_text(json.dumps(notes))

    return "\n".join(f"Part {i + 1}: {n}" for i, n in enumerate(notes))


def generate_style_guide(script_text, workdir):
    """
    Reads the WHOLE script -- no matter how long -- and generates a visual
    style brief that acts as the PARENT for every sentence's image prompt:
    the concrete topic, the mood, a specific color palette tailored to an
    American audience, and what real American settings fit the story. Every
    sentence prompt later inherits from this, the same way a child class
    inherits from a parent class, so every image in the video feels like it
    belongs to the same piece rather than being generated in isolation.

    This function is fully generic -- it contains no hardcoded topic,
    channel, or subject matter. It works the same way whether the script is
    about food safety, interest rates, or anything else.

    Cached to disk (resumable like every other stage). Does NOT silently
    fall back to a generic default -- if generation fails, that's a real
    problem worth surfacing and fixing, not papering over.
    """
    cache_path = workdir / "style_guide.txt"
    if cache_path.exists():
        print(f"[0/6] Style guide already exists, skipping: {cache_path}")
        return cache_path.read_text(encoding="utf-8")

    print("[0/6] Reading the full script and generating a visual style guide...")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print(
            "\nSTOPPED: GEMINI_API_KEY is not set, so a style guide can't be generated.\n"
            "Set it with: export GEMINI_API_KEY=\"your-key-here\"\n"
            "Then run this exact same command again -- narration, audio, and captions\n"
            "are already saved and will be reused automatically.\n"
        )
        sys.exit(1)

    try:
        script_understanding = read_full_script(script_text, workdir, gemini_key)

        prompt = (
            "You are a visual style consultant for a YouTube documentary channel made "
            "for an American audience. Below is either the full script, or notes "
            "covering the full script part-by-part (if it was long). Read all of it -- "
            "it spans the entire video, not just the beginning -- and understand the "
            "concept, the topic, and the mood it is aiming to deliver. Then write a "
            "visual style brief (under 200 words) that will act as the BASE STYLE every "
            "single shot in this video must inherit from. Structure it exactly like this:\n\n"
            "TOPIC: (3-8 words naming the concrete subject matter of this whole video)\n"
            "MOOD: (the overall emotional tone/mood of the video)\n"
            "COLOR PALETTE: (2-3 specific colors that fit this topic and an American "
            "visual-storytelling sensibility)\n"
            "SETTINGS: (the kind of real, everyday American locations/settings that fit "
            "this story -- specific, not abstract or generic)\n"
            "STYLE RULES: (how every shot should look and feel, and what to explicitly "
            "avoid -- it must read as authentically American and photorealistic, never "
            "like typical AI-generated art: no glossy/uncanny lighting, no surreal "
            "composition, no digital illustration look, no fashion photography, no "
            "character portraits of models unless a specific real person is named in the "
            "script)\n\n"
            "Output ONLY the brief in that structure, no preamble.\n\n"
            f"SCRIPT / SCRIPT NOTES:\n{script_understanding}"
        )

        generated = call_llm(prompt, gemini_key, max_tokens=350, temperature=0.7, min_content_length=300)

        if len(generated) > 20:
            print(f"      Style guide generated ({len(generated)} chars)")
            cache_path.write_text(generated, encoding="utf-8")
            return generated
        else:
            print(
                f"\nSTOPPED: the style guide came back too short/empty ({len(generated)} chars) "
                f"to be usable.\nThis usually means the model returned something unexpected -- "
                f"worth checking manually before continuing.\nRaw response: {generated!r}\n"
                f"Run this exact same command again once you've investigated -- everything\n"
                f"completed so far is saved and will be reused automatically.\n"
            )
            sys.exit(1)

    except SystemExit:
        raise
    except Exception as e:
        print(
            f"\nSTOPPED: style guide generation failed with an error: {e}\n"
            f"This needs to be looked at rather than silently working around it.\n"
            f"Run this exact same command again once it's fixed -- everything completed\n"
            f"so far (narration, audio, captions) is saved and will be reused automatically.\n"
        )
        sys.exit(1)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def _describe_google_error(e):
    """
    urllib only gives us 'HTTP Error 400: Bad Request' by default, which
    hides the actually useful part. Google's error responses are JSON with a
    real explanation (e.g. "API key not valid", "User location is not
    supported", quota details) -- read the body and surface that instead.
    """
    import urllib.error

    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            message = parsed.get("error", {}).get("message", body)
            return f"HTTP {e.code}: {message}"
        except Exception:
            return f"HTTP {e.code}: {e.reason}"
    return str(e)


_GEMINI_TEXT_MODEL_CACHE = None  # fetched once per run, reused by every call_llm() call


def _parse_quota_signal(error_text):
    """
    Google's 429 responses aren't just 'rate limited' -- they include the
    actual quota metric and its limit, and often a precise retry delay. Two
    very different situations produce the same HTTP 429:
      - "limit: 20" (or any N > 0): a real, small quota that WILL refill --
        worth a short wait.
      - "limit: 0": this model has NO free-tier quota on this key AT ALL.
        No amount of waiting fixes that -- it needs to be permanently
        skipped, not retried.
    Returns (limit_is_zero: bool, retry_after_seconds: float | None).
    """
    limit_is_zero = bool(re.search(r"limit:\s*0\b", error_text))
    retry_match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
    retry_after = float(retry_match.group(1)) if retry_match else None
    return limit_is_zero, retry_after


_PERMANENTLY_UNAVAILABLE_MODELS = set()  # models confirmed limit:0 -- never retry these again this run


def _get_gemini_text_candidates(api_key):
    """
    Asks Google for the CURRENT list of Gemini models available to this key
    and picks out text-capable ones (Flash-class, since that's what's free),
    fastest/cheapest first. Cached in-memory so this only hits the network
    once per run of the pipeline, not once per sentence.

    Falls back to a hardcoded list (current as of writing, per Google's live
    rate-limits page) only if the live lookup itself fails.
    """
    global _GEMINI_TEXT_MODEL_CACHE
    if _GEMINI_TEXT_MODEL_CACHE is not None:
        return _GEMINI_TEXT_MODEL_CACHE

    import urllib.request

    fallback = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read())

        text_ids = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            lname = name.lower()
            if (
                "generateContent" in methods
                and "flash" in lname
                and "image" not in lname
                and "embedding" not in lname
                and "tts" not in lname
                and "vision" not in lname
            ):
                text_ids.append(name)

        if text_ids:
            # Prefer plain "flash" over "flash-lite"/"flash-8b" variants by
            # sorting shorter names first (usually the more capable default).
            # Prefer newer 3.x Flash models over 2.x -- Google has started
            # telling new users 2.5-flash is deprecated in favor of 3.x, and
            # among same-generation options a shorter plain "flash" name
            # (vs "flash-lite") is usually the more capable default.
            def _version_rank(name):
                m = re.search(r"gemini-(\d+)", name)
                major = int(m.group(1)) if m else 0
                return (-major, len(name))
            text_ids.sort(key=_version_rank)
            _GEMINI_TEXT_MODEL_CACHE = text_ids[:5]
            print(f"      (using {len(_GEMINI_TEXT_MODEL_CACHE)} Gemini text models currently live for this key: {_GEMINI_TEXT_MODEL_CACHE})")
            return _GEMINI_TEXT_MODEL_CACHE

    except Exception as e:
        print(f"      (couldn't fetch the live Gemini model list: {_describe_google_error(e)} -- using a fallback list)")

    _GEMINI_TEXT_MODEL_CACHE = fallback
    return _GEMINI_TEXT_MODEL_CACHE


def _thinking_config_for_model(model_name):
    """
    Gemini 3.x and 2.5 models "think" before answering by default, and those
    hidden thinking tokens are deducted from maxOutputTokens -- exactly the
    bug that truncated responses down to a few words. Turn it down as much
    as each model generation actually allows:
      - Gemini 3.x: thinking CANNOT be fully disabled (per Google's own
        docs), so use the lowest level ("low") to minimize the burn.
      - Gemini 2.5: thinking CAN be fully disabled via thinkingBudget=0.
      - Gemini 2.0 and earlier: no thinking config exists, so return None.
    thinkingLevel and thinkingBudget must never both be set (that's a 400).
    """
    lname = model_name.lower()
    if lname.startswith("gemini-3"):
        return {"thinkingLevel": "low"}
    if lname.startswith("gemini-2.5"):
        return {"thinkingBudget": 0}
    return None


_GEMINI_TEXT_ROTATION_INDEX = 0  # spreads calls across models instead of exhausting one


def call_llm(prompt, api_key, max_tokens=200, temperature=0.7, min_content_length=20):
    """
    Shared helper: calls Gemini (Google AI Studio / Gemini Developer API) for
    text generation, trying free-tier Flash models.

    Free-tier daily/per-minute quotas on some models (especially newer
    preview ones) can be very small -- as low as ~20 requests. Two things
    follow from that:
      1. Retrying a 429 on the SAME model wastes time and doesn't help --
         each retry is itself another request against that same exhausted
         quota, digging the hole deeper instead of recovering. So a 429
         moves straight to the next model, no backoff loop.
      2. Calls round-robin across all available candidate models (starting
         from a different model each time this is called) instead of always
         hammering candidates[0] first -- this spreads 88 sentences' worth
         of calls across 5 models instead of exhausting the first one alone.
    Only raises if every candidate fails.
    """
    global _GEMINI_TEXT_ROTATION_INDEX
    import time
    import urllib.error
    import urllib.request

    all_candidates = _get_gemini_text_candidates(api_key)
    all_candidates = [m for m in all_candidates if m not in _PERMANENTLY_UNAVAILABLE_MODELS] or all_candidates
    idx = _GEMINI_TEXT_ROTATION_INDEX % len(all_candidates)
    candidates = all_candidates[idx:] + all_candidates[:idx]
    _GEMINI_TEXT_ROTATION_INDEX += 1

    all_errors = []
    saw_bad_key = False
    for model in candidates:
        thinking_config = _thinking_config_for_model(model)
        # Even at the lowest thinking level, some tokens still go to hidden
        # thinking before the real answer starts -- give real headroom so
        # the actual response isn't the part that gets cut off.
        effective_max_tokens = max_tokens * 6 if thinking_config else max_tokens

        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            req = urllib.request.Request(
                url,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            generation_config = {
                "maxOutputTokens": effective_max_tokens,
                "temperature": temperature,
            }
            if thinking_config:
                generation_config["thinkingConfig"] = thinking_config
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            })
            req.data = payload.encode("utf-8")
            with urllib.request.urlopen(req, timeout=90) as response:
                result = json.loads(response.read())

            candidates_out = result.get("candidates", [])
            content = ""
            if candidates_out:
                parts = candidates_out[0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts).strip()

            if not content:
                finish_reason = candidates_out[0].get("finishReason") if candidates_out else "NO_CANDIDATES"
                msg = f"model '{model}' returned empty content (finishReason: {finish_reason})"
                print(f"      [{msg} -- trying next]")
                all_errors.append(msg)
                continue
            if len(content) < min_content_length:
                msg = (
                    f"model '{model}' returned a suspiciously short/degenerate "
                    f"response ({len(content)} chars): {content!r}"
                )
                print(f"      [{msg} -- trying next]")
                all_errors.append(msg)
                continue

            # Small pacing delay after every successful call, to stay
            # comfortably under the free tier's requests-per-minute cap
            # instead of only reacting to 429s after they happen.
            time.sleep(2)
            return content

        except urllib.error.HTTPError as e:
            described = _describe_google_error(e)
            if e.code == 429:
                limit_is_zero, _ = _parse_quota_signal(described)
                if limit_is_zero:
                    print(f"      [model '{model}' has ZERO free-tier quota on this key -- permanently skipping it]")
                    _PERMANENTLY_UNAVAILABLE_MODELS.add(model)
                else:
                    print(f"      [model '{model}' rate-limited/quota-exhausted (429) -- moving to next model]")
            else:
                print(f"      [model '{model}' failed: {described} -- trying next]")
            all_errors.append(f"model '{model}' failed: {described}")
            if e.code == 400 and "api key" in described.lower():
                saw_bad_key = True
            continue

        except Exception as e:
            msg = f"model '{model}' failed: {e}"
            print(f"      [{msg} -- trying next]")
            all_errors.append(msg)
            continue

    if saw_bad_key:
        print(
            "\n      NOTE: Gemini says the API key itself isn't valid. Double-check\n"
            "      GEMINI_API_KEY was copied in full (no missing/extra characters) from\n"
            "      https://aistudio.google.com/apikey, and that it's exported in THIS\n"
            "      terminal session (re-export it if you opened a new terminal).\n"
        )

    raise RuntimeError("All Gemini text model candidates failed:\n  - " + "\n  - ".join(all_errors))


def generate_image_prompts(sentences, style_guide, script_text, workdir):
    cache_path = workdir / "image_prompts.json"
    all_prompts = []
    
    if cache_path.exists():
        existing = json.loads(cache_path.read_text())
        if len(existing) >= len(sentences):
            print(f"[0.5/6] Per-sentence image prompts already exist, skipping: {cache_path}")
            return existing[:len(sentences)]
        else:
            print(f"[0.5/6] Resuming from sentence {len(existing)+1}...")
            all_prompts = existing

    print(f"[0.5/6] Generating individual image prompts for {len(sentences)} sentences...")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("\nSTOPPED: GEMINI_API_KEY is not set.\n")
        sys.exit(1)

    # Loop sentence by sentence -- the response IS the string!
    for i in range(len(all_prompts), len(sentences)):
        sentence = sentences[i]
        prev_sentence = sentences[i - 1] if i > 0 else None
        next_sentence = sentences[i + 1] if i < len(sentences) - 1 else None
        print(f"      Sentence {i+1}/{len(sentences)}...")

        context_lines = []
        if prev_sentence:
            context_lines.append(f"PREVIOUS SENTENCE (for context only, do not depict this one): {prev_sentence}")
        context_lines.append(f"CURRENT SENTENCE (this is the one to depict): {sentence}")
        if next_sentence:
            context_lines.append(f"NEXT SENTENCE (for context only, do not depict this one): {next_sentence}")
        local_context = "\n".join(context_lines)

        # The style guide is the PARENT every sentence prompt must inherit from --
        # same topic, mood, color palette, settings, and style rules on every shot,
        # so the sentence prompt below only has to decide the SPECIFIC shot.
        prompt = (
            "You are a cinematographer writing a single detailed shot description for a "
            "photorealistic documentary-style video. This must work for any script on any "
            "topic -- do not assume a specific subject beyond what is given below.\n\n"
            f"STYLE GUIDE (the base every shot must inherit from -- topic, mood, color "
            f"palette, settings, and style rules):\n{style_guide}\n\n"
            f"SCRIPT CONTEXT (so this shot flows naturally from what came before and into "
            f"what comes next):\n{local_context}\n\n"
            "First, understand what the CURRENT SENTENCE is actually trying to convey "
            "visually -- the real-world action, object, place, or idea behind the words, "
            "not just the literal nouns. Then write ONE detailed, specific image-generation "
            "prompt for that single shot. It must:\n"
            "- Follow the style guide's palette, mood, and settings exactly (inherit, don't "
            "contradict it)\n"
            "- Specify a camera angle and shot type (e.g. low angle, eye-level, over-the-"
            "shoulder, close-up, wide establishing shot, macro detail shot)\n"
            "- Specify camera/lens settings that fit the shot (e.g. 35mm lens, shallow depth "
            "of field, natural window light, handheld, tripod-locked)\n"
            "- Depict a real, everyday American setting and be visually relatable to an "
            "American audience\n"
            "- Directly and specifically visualize what this sentence is saying -- not a "
            "vague or generic scene\n"
            "- Read as real photojournalism, never as AI-generated art (no glossy/uncanny "
            "lighting, no surreal composition, no illustration look)\n\n"
            "Output ONLY the finished prompt itself, nothing else -- no quotes, no labels, "
            "no preamble, no explanation of your choices."
        )

        final_prompt = None
        for attempt in range(1, 3):
            try:
                raw = call_llm(prompt, gemini_key, max_tokens=180, temperature=0.6, min_content_length=60)
                # Clean up any accidental wrapping quotes or markdown
                cleaned = raw.strip().strip('"').strip("'")
                if len(cleaned) > 10:
                    final_prompt = cleaned
                    break
            except Exception:
                pass

        # Ultimate fallback: if LLM fails for this sentence, use the sentence itself
        if not final_prompt:
            final_prompt = sentence

        all_prompts.append(final_prompt)
        # Save progress incrementally so you never lose work
        cache_path.write_text(json.dumps(all_prompts))

    print(f"      Successfully generated all {len(all_prompts)} prompts sentence-by-sentence.")
    return all_prompts

_GEMINI_IMAGE_MODEL_CACHE = None  # fetched once per run, reused by every call


def _get_gemini_image_candidates(api_key):
    """
    Same live-discovery approach as _get_gemini_text_candidates, but for
    image-generation-capable models (Nano Banana / Gemini image models).
    """
    global _GEMINI_IMAGE_MODEL_CACHE
    if _GEMINI_IMAGE_MODEL_CACHE is not None:
        return _GEMINI_IMAGE_MODEL_CACHE

    import urllib.request

    # Current as of writing, per Google's own rate-limits page: the stable
    # Nano Banana image model, falling back to older/preview naming.
    fallback = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
        "gemini-2.0-flash-preview-image-generation",
    ]

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read())

        image_ids = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            lname = name.lower()
            if "generateContent" in methods and "image" in lname and "preview" not in lname:
                image_ids.append(name)
        if not image_ids:
            # Some accounts only expose the preview naming -- accept that too
            # as a second pass rather than going straight to the hardcoded list.
            for m in data.get("models", []):
                name = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods and "image" in name.lower():
                    image_ids.append(name)

        if image_ids:
            image_ids.sort(key=len)
            _GEMINI_IMAGE_MODEL_CACHE = image_ids[:3]
            print(f"      (using {len(_GEMINI_IMAGE_MODEL_CACHE)} Gemini image models currently live for this key: {_GEMINI_IMAGE_MODEL_CACHE})")
            return _GEMINI_IMAGE_MODEL_CACHE

    except Exception as e:
        print(f"      (couldn't fetch the live Gemini image-model list: {_describe_google_error(e)} -- using a fallback list)")

    _GEMINI_IMAGE_MODEL_CACHE = fallback
    return _GEMINI_IMAGE_MODEL_CACHE


def generate_image_for_sentence(authored_prompt, out_path, width, height, api_key):
    """
    Returns the path on success, or None if every real option failed.
    NEVER writes a placeholder -- a placeholder in the final video is
    treated as unacceptable output, not a fallback.

    `authored_prompt` comes from generate_image_prompts() -- a specific,
    contextual prompt written by an LLM that read the whole script, not a
    generic template. We still append a technical/safety suffix here as
    cheap defense-in-depth, but the creative content is the LLM's.

    Uses Gemini's image-generation models (Nano Banana family) via the same
    API key as text generation. Nano Banana has its own "images per minute"
    (IPM) limit on the free tier, separate from and usually stricter than the
    text RPM limit -- so on a 429 we back off and retry the SAME model
    (an IPM limit clears within the minute) before falling through to a
    different model.
    """
    import base64
    import time
    import urllib.error
    import urllib.request

    prompt = (
        f"{authored_prompt} "
        f"photorealistic, shot on DSLR camera, natural lighting, real photography, "
        f"no illustration, no digital art, no CGI, no text, no watermark, "
        f"no fashion photography, no character portrait, no model posing, "
        f"no stylized rendering, editorial photojournalism only. "
        f"Aspect ratio matching {width}x{height}."
    )

    candidates = [m for m in _get_gemini_image_candidates(api_key) if m not in _PERMANENTLY_UNAVAILABLE_MODELS]

    for model in candidates:
        for attempt in range(2):  # a couple of tries per model, only when the wait is worth it
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={api_key}"
                )
                req = urllib.request.Request(
                    url,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                payload = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["IMAGE"]},
                })
                req.data = payload.encode("utf-8")
                with urllib.request.urlopen(req, timeout=90) as response:
                    result = json.loads(response.read())

                image_bytes = None
                for cand in result.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        inline = part.get("inlineData") or part.get("inline_data")
                        if inline and inline.get("data"):
                            image_bytes = base64.b64decode(inline["data"])
                            break
                    if image_bytes:
                        break

                if image_bytes and len(image_bytes) > 5_000:
                    with open(out_path, "wb") as f:
                        f.write(image_bytes)
                    # Pace proactively -- free-tier Nano Banana's images-per-
                    # minute limit is tight, so a fixed gap between successful
                    # generations avoids tripping it in the first place rather
                    # than only reacting after a 429.
                    time.sleep(8)
                    return out_path
                else:
                    print(f"      [Gemini model '{model}' returned no usable image data -- trying next]")
                    break

            except urllib.error.HTTPError as e:
                described = _describe_google_error(e)
                if e.code == 429:
                    limit_is_zero, retry_after = _parse_quota_signal(described)
                    if limit_is_zero:
                        # Not a rate limit -- this model has NO quota on this
                        # key at all. Waiting can never fix that. Blacklist it
                        # permanently for the rest of this run and move on
                        # immediately, no sleep wasted.
                        print(f"      [model '{model}' has ZERO free-tier quota on this key -- permanently skipping it]")
                        _PERMANENTLY_UNAVAILABLE_MODELS.add(model)
                        break
                    if attempt == 0 and retry_after is not None and retry_after < 30:
                        # A real quota that will genuinely refill soon --
                        # Google told us exactly how long, so wait that long,
                        # not a guessed number.
                        wait = retry_after + 1
                        print(f"      [model '{model}' rate-limited -- Google says retry in {retry_after:.0f}s, waiting {wait:.0f}s...]")
                        time.sleep(wait)
                        continue
                print(f"      [Gemini image model '{model}' failed: {described} -- trying next]")
                break

            except Exception as e:
                print(f"      [Gemini image model '{model}' failed: {e} -- trying next]")
                break

    # --- Every Gemini image model failed or is permanently unavailable on
    # this key. Try Pollinations.ai as a genuinely independent second
    # provider -- no key, no shared quota with Gemini, so it's not just
    # "the same failure again." ---
    try:
        import urllib.parse

        encoded = urllib.parse.quote(prompt)
        poll_url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
        req = urllib.request.Request(
            poll_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        if len(data) > 10_000:
            with open(out_path, "wb") as f:
                f.write(data)
            time.sleep(5)
            return out_path
        else:
            print(f"      [Pollinations fallback returned suspiciously small data ({len(data)} bytes)]")
    except Exception as e:
        print(f"      [Pollinations fallback also failed: {e}]")

    # --- Every real option failed. Do NOT create a placeholder. ---
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
    caption_y = int(height * 0.72)  # moved up from 0.80 for a more eye-friendly position
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
    timings_cache = workdir / "timings.json"
    if timings_cache.exists():
        print(f"[4/6] Caption timings already exist, skipping transcription: {timings_cache}")
        timings = json.loads(timings_cache.read_text())
    else:
        timings = get_word_timings(final_audio)
        timings_cache.write_text(json.dumps(timings))

    # --- Step 0 (runs here, before images): generate the video's style guide ---
    sentences = split_sentences(script_text)
    style_guide = generate_style_guide(script_text, workdir)

    # --- Step 0.5: write one contextual, detailed prompt PER SENTENCE ---
    image_prompts = generate_image_prompts(sentences, style_guide, script_text, workdir)

    # --- Step 5: images, RESUMABLE per-sentence -- and STOPS (no placeholder) on failure ---
    audio_clip = AudioFileClip(str(final_audio))
    duration = audio_clip.duration
    total_words = sum(len(s.split()) for s in sentences) or 1

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print(
            "\nSTOPPED: GEMINI_API_KEY is not set, so images can't be generated.\n"
            "Set it with: export GEMINI_API_KEY=\"your-key-here\"\n"
            "Then run this exact same command again -- narration, audio, captions,\n"
            "the style guide, and all written prompts are already saved and will be\n"
            "reused automatically.\n"
        )
        sys.exit(1)

    print(f"[5/6] Generating {len(sentences)} images with Ken Burns motion...")
    beat_clips = []
    t_cursor = 0.0
    pans = ["left", "right", "up", "down"]
    beat_targets = []  # (index, sentence, beat_duration) -- compute all targets first
    for i, sentence in enumerate(sentences):
        weight = len(sentence.split()) / total_words
        beat_duration = max(1.0, duration * weight)
        if t_cursor + beat_duration > duration:
            beat_duration = max(0.5, duration - t_cursor)
        beat_targets.append((i, sentence, beat_duration))
        t_cursor += beat_duration
        if t_cursor >= duration:
            break

    CROSSFADE = 0.5  # seconds of overlap between consecutive images
    n_beats = len(beat_targets)
    for idx, (i, sentence, beat_duration) in enumerate(beat_targets):
        img_path = workdir / f"beat_{i:03d}.png"
        if img_path.exists():
            print(f"      [{i+1}/{len(sentences)}] already generated, skipping: {img_path.name}")
        else:
            print(f"      [{i+1}/{len(sentences)}] generating image for sentence {i+1}...")
            result = generate_image_for_sentence(image_prompts[i], img_path, width, height, gemini_key)
            if result is None:
                print(
                    f"\nSTOPPED at sentence {i+1}/{len(sentences)}: every Gemini image model "
                    f"failed to generate a real image for this sentence.\n"
                    f"No placeholder was used -- nothing unusable will end up in your video.\n"
                    f"Everything completed so far (narration, music, captions, and "
                    f"{idx} earlier images) is saved in '{args.workdir}/' and will be reused automatically.\n"
                    f"Just run this exact same command again in a little while -- it will pick up\n"
                    f"right here at sentence {i+1} instead of starting over.\n"
                )
                sys.exit(1)

        # Add back the time the crossfade will consume, so the FINAL visible
        # duration on screen still equals beat_duration -- this is what keeps
        # total visuals in sync with the full audio length, no trailing blank gap.
        extra = CROSSFADE if idx < n_beats - 1 else 0
        clip_duration = beat_duration + extra
        beat_clips.append(
            ken_burns_clip(img_path, clip_duration, width, height, zoom_in=(i % 2 == 0), pan=pans[i % len(pans)])
        )

    # --- Step 6: assemble with crossfade transitions between beats ---
    print("[6/6] Compositing captions and rendering final video...")
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
