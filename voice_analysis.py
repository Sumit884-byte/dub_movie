"""Estimate voice pitch from Japanese dialogue audio."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from character_context import (
    _clamp_int,
    clamp_tts_profile,
    format_pitch_hz,
    format_rate_pct,
    infer_tone,
    is_female_voice,
    merge_tts_with_character,
    parse_pitch_hz,
    parse_rate_pct,
    VOICE_QUALITY_LIMITS,
)
from progress_eta import PhaseTimer, SEC_PITCH_PASS_LINE

# Screams / vocal SFX often read 450–600 Hz and push TTS pitch too high.
_F0_SCREAM_CEILING_HZ = 420


def extract_dialogue_mono(
    video_path: Path,
    wav_path: Path,
    duration: float | None = None,
    force: bool = False,
) -> Path:
    """Isolate center/dialogue channel for vocal removal and pitch readings."""
    if not force and wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-af", "pan=mono|c0=0.5*FC+0.35*FL+0.35*FR",
        "-ar", "44100", "-ac", "1",
    ]
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.append(str(wav_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def _load_slice(wav_path: Path, start: float, end: float, sr: int = 22050) -> np.ndarray | None:
    offset = max(0.0, start)
    duration = max(0.08, end - start)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(offset),
            "-t", str(duration),
            "-i", str(wav_path),
            "-ar", str(sr),
            "-ac", "1",
            tmp.name,
        ]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            return None
        samples, _file_sr = librosa.load(tmp.name, sr=sr, mono=True)
    if samples.size < sr * 0.05:
        return None
    return samples


def estimate_median_pitch_hz(samples: np.ndarray, sr: int = 22050) -> float | None:
    """Return median voiced F0 in Hz, or None if no voice detected."""
    if samples is None or samples.size == 0:
        return None

    f0 = librosa.yin(
        samples,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    voiced = f0[(f0 > 0) & np.isfinite(f0)]
    if voiced.size < 3:
        return None
    return float(np.median(voiced))


def analyze_segment_pitch(
    wav_path: Path,
    start: float,
    end: float,
) -> float | None:
    samples = _load_slice(wav_path, start, end)
    if samples is None:
        return None
    return estimate_median_pitch_hz(samples)


def profile_from_pitch(
    median_hz: float | None,
    speaker: str | None,
    base_profile: dict,
) -> dict:
    """
    Nudge character baseline pitch/rate from detected F0 — small deltas only.
    Japanese anime voices run high; Hindi TTS stacks badly if we add +55 Hz on top.
    """
    base = dict(base_profile)
    voice = base.get("voice", "hi-IN-MadhurNeural")
    base_pitch = parse_pitch_hz(base.get("pitch"))
    base_rate = parse_rate_pct(base.get("rate"))

    if median_hz is None or median_hz > _F0_SCREAM_CEILING_HZ:
        return clamp_tts_profile(base, base)

    female = is_female_voice(voice)
    d_pitch_lo, d_pitch_hi = VOICE_QUALITY_LIMITS["f0_pitch_delta"]
    d_rate_lo, d_rate_hi = VOICE_QUALITY_LIMITS["f0_rate_delta"]
    if female:
        delta = _clamp_int(int((median_hz - 300) * 0.05), d_pitch_lo, d_pitch_hi)
        gender = "female"
    else:
        delta = _clamp_int(int((median_hz - 200) * 0.04), d_pitch_lo, d_pitch_hi)
        if median_hz >= 240:
            gender = "child_male"
        elif median_hz >= 190:
            gender = "youth_male"
        else:
            gender = "male"

    pitch_hz = base_pitch + delta
    rate_pct = base_rate
    if female:
        if median_hz >= 340:
            rate_pct += _clamp_int(2, d_rate_lo, d_rate_hi)
        elif median_hz < 260:
            rate_pct += _clamp_int(-2, d_rate_lo, d_rate_hi)
    else:
        if median_hz >= 260:
            rate_pct += _clamp_int(3, d_rate_lo, d_rate_hi)
        elif median_hz < 170:
            rate_pct += _clamp_int(-2, d_rate_lo, d_rate_hi)

    profile = {
        "voice": voice,
        "rate": format_rate_pct(rate_pct),
        "pitch": format_pitch_hz(pitch_hz, voice=voice),
        "volume": base.get("volume", "+0%"),
        "median_f0_hz": round(median_hz, 1),
        "voice_gender": gender,
    }
    return clamp_tts_profile(profile, base)


def attach_pitch_to_segments(segments: list[dict], dialogue_wav: Path) -> list[dict]:
    """Attach median pitch per segment without building TTS profiles."""
    timer = PhaseTimer(
        "Pitch probe",
        len(segments),
        fallback_rate=1.0 / SEC_PITCH_PASS_LINE,
    )
    timer.announce()
    results = []
    for seg in segments:
        pitch_start = seg.get("srt_start", seg["start"])
        pitch_end = seg.get("srt_end", seg["end"])
        median_hz = analyze_segment_pitch(dialogue_wav, pitch_start, pitch_end)
        item = dict(seg)
        if median_hz is not None:
            item["median_f0_hz"] = round(median_hz, 1)
        results.append(item)
        timer.tick()
    timer.finish()
    pitched = sum(1 for seg in results if seg.get("median_f0_hz") is not None)
    print(f"Pitch probe: {pitched}/{len(results)} segments have F0 readings.")
    return results


def analyze_voices_for_segments(
    segments: list[dict],
    dialogue_wav: Path,
    base_profile_fn,
    language: str = "hi",
) -> list[dict]:
    print("Analyzing Japanese voice pitch per line...")
    timer = PhaseTimer(
        "Pitch analysis",
        len(segments),
        fallback_rate=1.0 / SEC_PITCH_PASS_LINE,
    )
    timer.announce()
    results = []
    for seg in segments:
        pitch_start = seg.get("srt_start", seg["start"])
        pitch_end = seg.get("srt_end", seg["end"])
        median_hz = analyze_segment_pitch(dialogue_wav, pitch_start, pitch_end)
        source = seg.get("source_ja") or seg["text"]
        tone = infer_tone(source, seg.get("speaker"))
        base = base_profile_fn(seg.get("speaker"), language)
        profile = profile_from_pitch(median_hz, seg.get("speaker"), base)
        profile = merge_tts_with_character(profile, seg.get("speaker"), tone, language)
        item = {**seg, "profile": profile, "tone": tone}
        if median_hz is not None:
            item["median_f0_hz"] = round(median_hz, 1)
        results.append(item)
        timer.tick()
    timer.finish()

    pitched = [s for s in results if s.get("median_f0_hz") is not None]
    if pitched:
        females = sum(1 for s in pitched if s["profile"].get("voice_gender") in ("female", "child_female"))
        males = sum(1 for s in pitched if s["profile"].get("voice_gender") == "male")
        print(
            f"Pitch analysis: {len(pitched)}/{len(results)} lines — "
            f"~{females} higher-pitch, ~{males} lower-pitch voices."
        )
    return results


VOICE_CACHE_VERSION = 1
_VOICE_CACHE_FIELDS = ("median_f0_hz", "tone", "profile", "speaker")


def save_voice_cache(path: Path, segments: list[dict]) -> None:
    """Persist pitch + TTS profiles so resume can skip pitch passes."""
    payload = {
        "version": VOICE_CACHE_VERSION,
        "entries": {
            seg["key"]: {field: seg.get(field) for field in _VOICE_CACHE_FIELDS}
            for seg in segments
            if seg.get("key")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_voice_cache(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("version") != VOICE_CACHE_VERSION:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return None
    return entries


def apply_voice_cache(segments: list[dict], path: Path) -> list[dict] | None:
    """Merge cached pitch/profiles onto fresh SRT segments, or None if cache is stale."""
    entries = load_voice_cache(path)
    if not entries:
        return None

    missing = [seg for seg in segments if seg.get("key") not in entries]
    if missing:
        print(
            f"Voice cache stale: {len(missing)}/{len(segments)} lines missing — "
            "recomputing pitch."
        )
        return None

    merged: list[dict] = []
    for seg in segments:
        item = dict(seg)
        cached = entries[seg["key"]]
        for field in _VOICE_CACHE_FIELDS:
            if field in cached and cached[field] is not None:
                item[field] = cached[field]
        merged.append(item)
    return merged


def voice_cache_entry_count(path: Path | None) -> int:
    if not path:
        return 0
    entries = load_voice_cache(path)
    return len(entries) if entries else 0


def apply_voice_from_translation_cache(
    segments: list[dict],
    cache_path: Path,
) -> list[dict] | None:
    """Restore pitch/TTS profiles embedded in a translation cache (legacy resume path)."""
    if not cache_path.exists():
        return None
    try:
        cached_list = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(cached_list, list):
        return None

    by_key = {
        item["key"]: item
        for item in cached_list
        if isinstance(item, dict) and item.get("key") and item.get("profile")
    }
    if not by_key:
        return None

    merged: list[dict] = []
    restored = 0
    for seg in segments:
        item = dict(seg)
        cached_seg = by_key.get(seg.get("key"))
        if cached_seg:
            for field in _VOICE_CACHE_FIELDS:
                if cached_seg.get(field) is not None:
                    item[field] = cached_seg[field]
            restored += 1
        merged.append(item)

    if restored < len(by_key):
        print(
            f"Translation cache voice data stale: {restored}/{len(by_key)} matched — "
            "recomputing pitch."
        )
        return None

    print(
        f"Restored pitch/TTS profiles from translation cache: {cache_path.name} "
        f"({restored} lines, skipping pitch probe + analysis)."
    )
    return merged


def load_voiced_segments(
    segments: list[dict],
    *,
    voice_cache_path: Path,
    translation_cache_path: Path | None = None,
) -> list[dict] | None:
    """Load pitch/TTS profiles from voice cache or translation cache."""
    voiced = apply_voice_cache(segments, voice_cache_path)
    if voiced is not None:
        print(
            f"Loaded voice/pitch profiles from cache: {voice_cache_path.name} "
            f"({len(voiced)} lines, skipping pitch probe + analysis)."
        )
        return voiced

    if translation_cache_path:
        voiced = apply_voice_from_translation_cache(segments, translation_cache_path)
        if voiced is not None:
            save_voice_cache(voice_cache_path, voiced)
            print(f"Saved voice/pitch cache: {voice_cache_path.name}")
            return voiced

    return None
