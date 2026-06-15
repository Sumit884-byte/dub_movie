import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edge_tts
import numpy as np
from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip

from character_context import (
    CHARACTER_PROFILES,
    DEFAULT_CHILD_MALE,
    DEFAULT_FEMALE,
    DEFAULT_MALE,
    batch_contextual_translate,
    character_profile,
    clamp_tts_profile,
    contextual_translate,
    f0_matches_character,
    finalize_segment_speakers,
    is_vocal_filler,
    merge_tts_with_character,
    propagate_speaker,
    should_dub_segment,
)
from voice_analysis import attach_pitch_to_segments, extract_dialogue_mono, profile_from_pitch
from speaker_id_vllm import identify_speakers_vllm
from progress_eta import (
    PhaseTimer,
    SEC_RENDER_BASE,
    SEC_RENDER_PER_VIDEO_SEC,
    SEC_TTS_BATCH,
    SEC_TRANSLATE_LINE,
    format_eta,
)

SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
SPEAKER_TAG = re.compile(r"^（([^）]+)）")
SPEAKER_INLINE = re.compile(r"（([^）]+)）")
POSITION_TAG = re.compile(r"\{\\an(\d+)\}")
TOP_SCREEN_ALIGNMENTS = {7, 8, 9}

DEFAULT_WORKERS = max(2, min(8, os.cpu_count() or 4))
_cache_lock = threading.Lock()

# Final mix: keep full background bed; dub sits slightly above it.
BACKGROUND_MIX_GAIN = 1.5
DUB_MIX_GAIN = 1.25


def default_workers() -> int:
    return DEFAULT_WORKERS


def segment_key(seg: dict) -> str:
    return f"{seg['start']:.3f}_{seg['end']:.3f}"


def chunk_path(chunks_dir: Path, seg: dict) -> Path:
    speaker = re.sub(r"[^\w\u3040-\u30ff]+", "_", seg.get("speaker") or "unknown")
    return chunks_dir / f"chunk_{segment_key(seg)}_{speaker}.mp3"


def chunk_exists(path: Path, *, min_bytes: int = 256) -> bool:
    """Fast on-disk check — no ffprobe (safe for thousands of chunks)."""
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def chunk_is_valid(path: Path, *, min_bytes: int = 256) -> bool:
    """True when a TTS chunk looks like non-empty audio."""
    return chunk_exists(path, min_bytes=min_bytes)


class JobState:
    _STAGE_RANK = {
        "start": 0,
        "whisper_audio_ready": 1,
        "whisper_done": 2,
        "aligned": 3,
        "speakers_vllm": 4,
        "voice_analyzed": 5,
        "translating": 6,
        "translated": 7,
        "background_ready": 8,
        "synthesizing": 9,
        "synthesized": 10,
        "rendering": 11,
        "done": 12,
    }

    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"stage": "start", "updated_at": None}

    def save(self, **updates) -> None:
        new_stage = updates.get("stage")
        if new_stage is not None:
            current = self.data.get("stage", "start")
            if self._STAGE_RANK.get(new_stage, -1) < self._STAGE_RANK.get(current, -1):
                updates = {key: value for key, value in updates.items() if key != "stage"}
        self.data.update(updates)
        self.data["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def stage(self) -> str:
        return self.data.get("stage", "start")


def srt_timestamp(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def normalize_speaker(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name)
    return name.split("･")[0].split("・")[0].strip()


def extract_speaker(raw_text: str) -> str | None:
    for line in raw_text.splitlines():
        line = line.strip()
        match = SPEAKER_TAG.match(line)
        if match:
            return normalize_speaker(match.group(1))
        match = SPEAKER_INLINE.search(line)
        if match:
            name = normalize_speaker(match.group(1))
            if "声" not in name and "音" not in name:
                return name
    return None


def is_sfx_only(raw_text: str) -> bool:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return True
    return all(SPEAKER_TAG.fullmatch(line) or "音" in line or "鳴" in line for line in lines)


def extract_position(raw_text: str) -> int | None:
    match = POSITION_TAG.search(raw_text)
    if not match:
        return None
    return int(match.group(1))


def is_top_on_screen(raw_text: str) -> bool:
    pos = extract_position(raw_text)
    return pos in TOP_SCREEN_ALIGNMENTS if pos is not None else False


def profile_for_segment(
    seg: dict,
    language: str = "hi",
    *,
    prev_speaker: str | None = None,
) -> dict:
    tone = seg.get("tone", "neutral")
    speaker = seg.get("speaker")
    base = character_profile(speaker, language)
    f0 = seg.get("median_f0_hz")
    speaker_changed = bool(prev_speaker and speaker and prev_speaker != speaker)
    use_f0 = (
        f0 is not None
        and not speaker_changed
        and f0_matches_character(f0, speaker)
    )
    if use_f0:
        working = profile_from_pitch(f0, speaker, base)
    else:
        working = dict(base)
    return merge_tts_with_character(working, speaker, tone, language)


def voice_profile_for_speaker(speaker: str | None, language: str = "hi") -> dict:
    return character_profile(speaker, language)


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"\{[^}]+\}", "", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if SPEAKER_TAG.fullmatch(line):
            continue
        line = SPEAKER_TAG.sub("", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def parse_srt(path: Path, max_duration: float | None = None) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue

        time_match = None
        text_lines = []
        for line in lines:
            match = SRT_TIME.search(line)
            if match:
                time_match = match
            elif not line.strip().isdigit():
                text_lines.append(line)

        if not time_match:
            continue

        raw_text = "\n".join(text_lines)
        if is_sfx_only(raw_text):
            continue

        start = srt_timestamp(*time_match.groups()[:4])
        end = srt_timestamp(*time_match.groups()[4:])
        if max_duration is not None and start >= max_duration:
            break

        text = clean_subtitle_text(raw_text)
        if not text:
            continue

        if max_duration is not None:
            end = min(end, max_duration)

        segments.append(
            {
                "start": start,
                "end": end,
                "srt_start": start,
                "srt_end": end,
                "text": text,
                "source_ja": text,
                "speaker": extract_speaker(raw_text),
                "on_screen": is_top_on_screen(raw_text),
                "position": extract_position(raw_text),
                "key": None,
            }
        )

    for seg in segments:
        seg["key"] = segment_key(seg)
    return segments


def _sanitize_segments(segments: list[dict]) -> list[dict]:
    clean = []
    for seg in segments:
        text = seg.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text or "Error 500" in text or "That's an error" in text:
            continue
        item = {**seg, "text": text}
        item["key"] = item.get("key") or segment_key(item)
        clean.append(item)
    return sorted(clean, key=lambda s: s["start"])


def _write_cache(cache_path: Path, segments: list[dict]) -> None:
    with _cache_lock:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cache_path)


def _load_cache(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        return []
    return _sanitize_segments(json.loads(cache_path.read_text(encoding="utf-8")))


def _translate_one(
    seg: dict,
    target_language: str,
    *,
    translate_backend: str = "auto",
    translate_model: str | None = None,
) -> dict | None:
    source_ja = seg.get("source_ja") or seg["text"]
    speaker = seg.get("speaker")
    if is_vocal_filler(source_ja):
        return None
    for attempt in range(5):
        try:
            hindi_text, tone = contextual_translate(
                source_ja,
                speaker,
                target_language,
                translate_backend=translate_backend,
                translate_model=translate_model,
            )
            if not isinstance(hindi_text, str) or not hindi_text.strip():
                raise ValueError("empty translation")
            return {
                **{k: v for k, v in seg.items() if k != "text"},
                "source_ja": source_ja,
                "tone": tone,
                "text": hindi_text.strip(),
            }
        except Exception as exc:
            if attempt == 4:
                print(f"Skipping {seg['key']} after translation failed: {exc}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def translate_segments(
    segments: list[dict],
    target_language: str,
    cache_path: Path,
    workers: int,
    state: JobState,
    *,
    translate_backend: str = "auto",
    translate_model: str | None = None,
) -> list[dict]:
    cached = _load_cache(cache_path)
    done_keys = {seg["key"] for seg in cached}
    pending = [seg for seg in segments if seg["key"] not in done_keys]

    if cached and not pending:
        print(f"All {len(cached)} translations loaded from cache.")
        state.save(stage="translated", translated=len(cached))
        return cached

    if cached:
        print(f"Resuming translation: {len(cached)} done, {len(pending)} remaining.")
    else:
        print(
            f"Translating {len(segments)} lines via {translate_backend} "
            f"with context/tone ({workers} workers)..."
        )
    if pending:
        eta_sec = len(pending) * SEC_TRANSLATE_LINE
        print(f"Translation ETA: ~{format_eta(eta_sec)} ({len(pending)} lines pending)")

    segments = propagate_speaker(segments)

    translated = list(cached)
    state.save(stage="translating", translated=len(translated), total=len(segments))

    if pending:
        batch_results, failed = batch_contextual_translate(
            pending,
            target_language,
            translate_backend=translate_backend,
            workers=workers,
        )
        translated.extend(batch_results)
        ordered = _sanitize_segments(translated)
        _write_cache(cache_path, ordered)
        state.save(stage="translating", translated=len(ordered), total=len(segments))
        print(f"Translated {len(ordered)}/{len(segments)}")
        if failed:
            failed_path = cache_path.with_suffix(".failed.json")
            failed_path.write_text(
                json.dumps(failed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"WARNING: {len(failed)} line(s) failed all backends — "
                f"no Hindi dub for those slots. Saved to {failed_path.name}. "
                "Re-run the job later to retry them."
            )
            state.save(stage="translating", translated=len(ordered), failed=len(failed))

    ordered = _sanitize_segments(translated)
    _write_cache(cache_path, ordered)
    failed_path = cache_path.with_suffix(".failed.json")
    if failed_path.exists():
        failed_count = len(json.loads(failed_path.read_text(encoding="utf-8")))
        if failed_count:
            print(
                f"Translation complete: {len(ordered)} lines dubbed, "
                f"{failed_count} failed (see {failed_path.name})."
            )
        else:
            print(f"Translation complete: {len(ordered)} lines.")
    else:
        print(f"Translation complete: {len(ordered)} lines.")
    state.save(stage="translated", translated=len(ordered), total=len(segments))
    return ordered


async def _synthesize_chunk(text: str, output_path: Path, profile: dict) -> bool:
    """Synthesize one line; return False if Edge TTS fails after retries."""
    text = text.strip()
    if not text:
        return False

    voice = profile["voice"]
    attempts: list[dict] = [
        {
            "rate": profile.get("rate", "+0%"),
            "pitch": profile.get("pitch", "+0Hz"),
            "volume": profile.get("volume", "+0%"),
        },
        {
            "rate": "+0%",
            "pitch": "+0Hz",
            "volume": profile.get("volume", "+0%"),
        },
    ]

    for attempt, prosody in enumerate(attempts, start=1):
        safe = clamp_tts_profile({"voice": voice, **prosody})
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=safe["voice"],
                rate=safe.get("rate", "+0%"),
                pitch=safe.get("pitch", "+0Hz"),
                volume=safe.get("volume", "+0%"),
            )
            await communicate.save(str(output_path))
            if chunk_is_valid(output_path):
                return True
            output_path.unlink(missing_ok=True)
        except edge_tts.exceptions.NoAudioReceived as exc:
            output_path.unlink(missing_ok=True)
            if attempt >= len(attempts):
                print(f"TTS failed for {output_path.name!r}: {exc}")
            else:
                await asyncio.sleep(1.5 * attempt)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            print(f"TTS error for {output_path.name!r}: {exc}")
            await asyncio.sleep(1.5 * attempt)
    return False


async def _synthesize_all(
    segments: list[dict],
    chunks_dir: Path,
    workers: int,
    language: str,
    *,
    existing: int = 0,
) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(workers)
    failed: list[str] = []
    timer = PhaseTimer(
        "TTS",
        len(segments),
        initial=existing,
        fallback_rate=workers / SEC_TTS_BATCH,
    )
    timer.announce()

    async def run_one(seg: dict, previous: str | None) -> None:
        path = chunk_path(chunks_dir, seg)
        if chunk_exists(path):
            return
        if path.exists():
            path.unlink(missing_ok=True)
        profile = profile_for_segment(seg, language, prev_speaker=previous)
        async with sem:
            ok = await _synthesize_chunk(seg["text"], path, profile)
        if not ok:
            failed.append(seg.get("key", path.stem))
        timer.tick()

    await asyncio.gather(
        *(
            run_one(seg, segments[i - 1].get("speaker") if i else None)
            for i, seg in enumerate(segments)
        )
    )
    timer.finish()
    if failed:
        print(f"Warning: {len(failed)} TTS line(s) failed — those slots will be silent.")


MAX_DUB_SPEEDUP = 2.0


def fit_audio_to_slot(audio_clip, slot_duration: float):
    """Time-compress (pitch-preserving) and trim so dub stays within subtitle timing."""
    slot_duration = max(slot_duration, 0.1)
    if audio_clip.duration <= slot_duration:
        return audio_clip

    speed_factor = min(audio_clip.duration / slot_duration, MAX_DUB_SPEEDUP)
    if speed_factor > 1.02:
        import librosa
        from moviepy import AudioArrayClip

        fps = int(getattr(audio_clip, "fps", None) or 44100)
        samples = audio_clip.to_soundarray(fps=fps)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        stretched = librosa.effects.time_stretch(np.asarray(samples, dtype=np.float32), rate=speed_factor)
        max_samples = int(slot_duration * fps)
        if stretched.size > max_samples:
            stretched = stretched[:max_samples]
        audio_clip = AudioArrayClip(stretched.reshape(-1, 1), fps=fps)
    elif audio_clip.duration > slot_duration:
        audio_clip = audio_clip.subclipped(0, slot_duration)

    if audio_clip.duration > slot_duration:
        audio_clip = audio_clip.subclipped(0, slot_duration)
    return audio_clip


def build_dub_timeline_wav(
    segments: list[dict],
    chunks_dir: Path,
    output_wav: Path,
    duration: float,
) -> tuple[int, int]:
    """Mix per-line MP3 chunks into one WAV — one file open at a time."""
    import librosa
    import soundfile as sf

    fps = 44100
    total_samples = int(math.ceil(duration * fps)) + fps
    timeline = np.zeros(total_samples, dtype=np.float32)

    dub_segments = [seg for seg in segments if should_dub_segment(seg)]
    timer = PhaseTimer("Dub timeline", len(dub_segments), fallback_rate=40.0)
    timer.announce()

    used = 0
    skipped_bad = 0
    for seg in dub_segments:
        path = chunk_path(chunks_dir, seg)
        if not chunk_exists(path):
            if path.exists():
                path.unlink(missing_ok=True)
            print(f"Warning: missing chunk for {seg['key']}, skipping.")
            skipped_bad += 1
            timer.tick()
            continue

        slot_duration = max(seg["end"] - seg["start"], 0.1)
        try:
            samples, _ = librosa.load(str(path), sr=fps, mono=True)
        except Exception as exc:
            print(f"Warning: unreadable chunk for {seg['key']} ({exc}), skipping.")
            path.unlink(missing_ok=True)
            skipped_bad += 1
            timer.tick()
            continue

        sample_duration = len(samples) / fps
        if sample_duration > slot_duration:
            speed_factor = min(sample_duration / slot_duration, MAX_DUB_SPEEDUP)
            if speed_factor > 1.02:
                samples = librosa.effects.time_stretch(
                    np.asarray(samples, dtype=np.float32),
                    rate=speed_factor,
                )
            max_samples = int(slot_duration * fps)
            if len(samples) > max_samples:
                samples = samples[:max_samples]

        start = int(seg["start"] * fps)
        end = min(start + len(samples), len(timeline))
        count = end - start
        if count > 0:
            timeline[start:end] += samples[:count] * DUB_MIX_GAIN
            used += 1
        timer.tick()

    timer.finish()
    peak = float(np.max(np.abs(timeline))) or 1.0
    if peak > 1.0:
        timeline /= peak

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), timeline, fps)
    return used, skipped_bad


def synthesize_dubbed_audio(
    segments: list[dict],
    chunks_dir: Path,
    workers: int,
    state: JobState,
    target_language: str = "hi",
    *,
    timeline_duration: float,
) -> Path:
    segments = finalize_segment_speakers(segments)
    existing = sum(1 for seg in segments if chunk_exists(chunk_path(chunks_dir, seg)))
    remaining = len(segments) - existing
    if existing:
        print(f"Resuming TTS: {existing}/{len(segments)} chunks already on disk.")
    else:
        print(f"Synthesizing {len(segments)} lines with {workers} concurrent TTS tasks...")
    if remaining > 0:
        eta_sec = (remaining / max(1, workers)) * SEC_TTS_BATCH
        print(f"TTS ETA: ~{format_eta(eta_sec)} ({remaining} lines remaining)")

    state.save(stage="synthesizing", synthesized=existing, total=len(segments))
    asyncio.run(_synthesize_all(segments, chunks_dir, workers, target_language, existing=existing))

    dub_wav = chunks_dir / "dub_timeline.wav"
    used, skipped_bad = build_dub_timeline_wav(
        segments, chunks_dir, dub_wav, timeline_duration
    )
    if skipped_bad:
        print(f"Skipped {skipped_bad} invalid/missing TTS chunk(s) — re-run to retry them.")

    state.save(stage="synthesized", synthesized=used, total=len(segments))
    print(f"TTS timeline ready: {used} clips mixed -> {dub_wav.name}")
    return dub_wav


def _audio_channel_count(input_video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "csv=p=0",
            str(input_video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip() or "2")


def _background_audio_filter(channels: int, *, center_cut: float = 0.72) -> str:
    """Build a stereo bed that drops dialogue (center) but keeps music and SFX."""
    if channels >= 6:
        # 5.1: omit FC entirely; keep fronts, surrounds, and a touch of LFE.
        return (
            "pan=stereo|"
            "FL=0.95*FL+0.88*BL+0.14*LFE|"
            "FR=0.95*FR+0.88*BR+0.14*LFE"
        )
    if channels >= 3:
        # 3ch/4ch with explicit center: subtract dialogue from L/R only.
        return f"pan=stereo|FL=FL-{center_cut}*FC|FR=FR-{center_cut}*FC"
    # Plain stereo: subtract the mono (center) component from each channel.
    mid = f"{center_cut}*0.5*(FL+FR)"
    return f"pan=stereo|FL=FL-{mid}|FR=FR-{mid}"


def _subtract_vocals_from_bed(
    bed_wav: Path,
    vocal_wav: Path,
    output_wav: Path,
    strength: float = 0.55,
) -> None:
    """Remove a dialogue-heavy mono track from stereo without erasing the full mix."""
    import librosa
    import numpy as np
    from scipy.io import wavfile

    bed, _sr = librosa.load(str(bed_wav), sr=44100, mono=False)
    vox, _ = librosa.load(str(vocal_wav), sr=44100, mono=True)
    if bed.ndim == 1:
        bed = np.vstack([bed, bed])
    length = min(bed.shape[1], len(vox))
    bed = bed[:, :length].copy()
    vox_stereo = np.vstack([vox[:length], vox[:length]])
    cleaned = np.clip(bed - strength * vox_stereo, -1.0, 1.0)
    wavfile.write(str(output_wav), 44100, cleaned.T)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_stereo_audio(input_video: Path, output_wav: Path, force: bool = False) -> Path:
    if not force and output_wav.exists() and output_wav.stat().st_size > 0:
        return output_wav
    cmd = [
        "ffmpeg", "-y", "-i", str(input_video),
        "-vn", "-ac", "2", "-ar", "44100",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_wav


def extract_background_audio(
    input_video: Path,
    output_wav: Path,
    force: bool = False,
    dialogue_wav: Path | None = None,
    use_demucs: bool = False,
    center_cut: float = 0.72,
) -> Path:
    if not force and output_wav.exists() and output_wav.stat().st_size > 0:
        print(f"Using existing background audio: {output_wav}")
        return output_wav

    channels = _audio_channel_count(input_video)

    if use_demucs and channels < 6:
        try:
            from vocal_separation import separate_with_demucs

            source_wav = output_wav.with_suffix(".source_stereo.wav")
            extract_stereo_audio(input_video, source_wav, force=force)
            return separate_with_demucs(
                source_wav,
                output_wav,
                work_dir=output_wav.parent / "demucs_out",
                force=force,
            )
        except Exception as exc:
            print(f"Demucs unavailable ({exc}); falling back to center-channel removal.")

    # Drop dialogue from the center channel only — keep music, ambience, and SFX.
    audio_filter = _background_audio_filter(channels, center_cut=center_cut)
    if channels >= 6:
        print(f"Extracting 5.1 background ({channels}ch, FC/dialogue omitted, music+SFX kept)...")
    elif channels >= 3:
        print(f"Extracting background ({channels}ch, center dialogue reduced)...")
    else:
        print("Extracting stereo background (center/mono dialogue reduced)...")

    cmd = [
        "ffmpeg", "-y", "-i", str(input_video),
        "-af", audio_filter,
        "-ac", "2", "-ar", "44100",
        str(output_wav),
    ]
    if subprocess.run(cmd, capture_output=True, text=True).returncode != 0:
        # Last-resort fallback for odd layouts: light vocal subtraction, not full erase.
        if dialogue_wav and dialogue_wav.exists() and dialogue_wav.stat().st_size > 0:
            print("Channel routing failed; using light vocal subtraction fallback...")
            source_wav = output_wav.with_suffix(".full_stereo.wav")
            extract_stereo_audio(input_video, source_wav, force=force)
            _subtract_vocals_from_bed(source_wav, dialogue_wav, output_wav, strength=0.45)
            source_wav.unlink(missing_ok=True)
            return output_wav
        raise RuntimeError("Failed to extract background audio with ffmpeg")

    return output_wav


def prepare_input_video(input_video: Path, duration: float | None, force: bool = False) -> Path:
    if duration is None:
        return input_video

    clip_path = input_video.with_name(f"{input_video.stem}_{int(duration)}s{input_video.suffix}")
    if force and clip_path.exists():
        clip_path.unlink()

    if not clip_path.exists():
        print(f"Creating {duration}s clip from start: {clip_path}")
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", "0", "-i", str(input_video),
                "-t", str(duration), "-c", "copy", str(clip_path),
            ],
            check=True,
        )
    return clip_path


def master_final_video(
    input_video: Path,
    output_video: Path,
    background_wav: Path,
    workers: int,
    state: JobState,
    *,
    dub_wav: Path | None = None,
    dubbed_clips: list | None = None,
    force: bool = False,
) -> None:
    if not force and output_video.exists() and state.stage == "done":
        print(f"Output already complete: {output_video.resolve()}")
        return

    clip_duration = _probe_duration(input_video)
    render_eta = SEC_RENDER_BASE + clip_duration * SEC_RENDER_PER_VIDEO_SEC
    print(
        f"Mixing audio ({clip_duration:.3f}s) then muxing with original video "
        f"(stream copy — no video re-encode)... "
        f"ETA ~{format_eta(render_eta)}"
    )
    state.save(stage="rendering")

    if dub_wav is not None:
        if not dub_wav.exists():
            raise FileNotFoundError(f"Dub timeline missing: {dub_wav}")
        # Single ffmpeg pass: 5.1 bed + dub timeline -> AAC in MP4 (no MoviePy temp WAVs).
        filter_complex = (
            f"[1:a]volume={BACKGROUND_MIX_GAIN}[bg];"
            f"[2:a]volume={DUB_MIX_GAIN}[du];"
            f"[bg][du]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        out_tmp = output_video.with_suffix(".tmp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-i", str(background_wav),
            "-i", str(dub_wav),
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", f"{clip_duration:.6f}",
            "-movflags", "+faststart",
            str(out_tmp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            out_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg mux/mix failed:\n{result.stderr[-4000:]}")
        out_tmp.replace(output_video)
        state.save(stage="done", output=str(output_video.resolve()))
        print(f"Saved: {output_video.resolve()}")
        return

    temp_audio = output_video.with_suffix(".final_audio.wav")
    bg_temp = temp_audio.with_suffix(".bg.wav")
    dub_temp = temp_audio.with_suffix(".dub.wav")
    bg_cmd = [
        "ffmpeg", "-y",
        "-i", str(background_wav),
        "-af", f"volume={BACKGROUND_MIX_GAIN}",
        "-t", f"{clip_duration:.6f}",
        "-ar", "44100", "-ac", "2",
        str(bg_temp),
    ]
    bg_result = subprocess.run(bg_cmd, capture_output=True, text=True)
    if bg_result.returncode != 0:
        raise RuntimeError(f"ffmpeg background export failed:\n{bg_result.stderr[-2000:]}")

    if dubbed_clips:
        dubbed_dialogue = (
            CompositeAudioClip(dubbed_clips)
            .with_duration(clip_duration)
            .with_volume_scaled(DUB_MIX_GAIN)
        )
        dubbed_dialogue.write_audiofile(str(dub_temp), fps=44100, logger=None)
        dubbed_dialogue.close()
        for clip in dubbed_clips:
            try:
                clip.close()
            except OSError:
                pass
    else:
        raise RuntimeError("master_final_video requires dub_wav or dubbed_clips")

    mix_cmd = [
        "ffmpeg", "-y",
        "-i", str(bg_temp),
        "-i", str(dub_temp),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0",
        "-t", f"{clip_duration:.6f}",
        str(temp_audio),
    ]
    mix_result = subprocess.run(mix_cmd, capture_output=True, text=True)
    bg_temp.unlink(missing_ok=True)
    dub_temp.unlink(missing_ok=True)
    if mix_result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed:\n{mix_result.stderr[-2000:]}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-i", str(temp_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{clip_duration:.6f}",
        "-movflags", "+faststart",
        str(output_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-2000:]}")

    temp_audio.unlink(missing_ok=True)
    state.save(stage="done", output=str(output_video.resolve()))
    print(f"Saved: {output_video.resolve()}")


def dub_from_subtitles(
    input_video: Path,
    subtitle_file: Path,
    output_video: Path,
    target_language: str = "hi",
    duration: float | None = None,
    cache_path: Path | None = None,
    chunks_dir: Path | None = None,
    workers: int | None = None,
    state_path: Path | None = None,
    use_vllm_speakers: bool = False,
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_speaker_cache: Path | None = None,
    fresh: bool = False,
    translate_backend: str = "auto",
    translate_model: str | None = None,
) -> None:
    workers = workers or default_workers()
    cache_path = cache_path or output_video.with_suffix(".segments.json")
    chunks_dir = chunks_dir or Path("temp_chunks") / output_video.stem
    state_path = state_path or output_video.with_suffix(".state.json")
    state = JobState(state_path)

    print(f"Workers: {workers} | State file: {state_path}")

    segments = parse_srt(subtitle_file, max_duration=duration)
    if not segments:
        raise RuntimeError(f"No usable subtitle lines found in {subtitle_file}")

    print(f"Parsed {len(segments)} dialogue lines from {subtitle_file}")
    segments = propagate_speaker(segments)

    video_path = prepare_input_video(input_video, duration, force=fresh)
    dialogue_wav = video_path.with_suffix(".dialogue.wav")
    extract_dialogue_mono(video_path, dialogue_wav, duration=duration, force=fresh)

    if use_vllm_speakers:
        segments = attach_pitch_to_segments(segments, dialogue_wav)
        speaker_cache = vllm_speaker_cache or output_video.with_suffix(".speakers.json")
        segments, speaker_summary = identify_speakers_vllm(
            segments,
            cache_path=speaker_cache,
            base_url=vllm_base_url,
            model=vllm_model,
        )
        state.save(stage="speakers_vllm", speaker_summary=speaker_summary)

    translated = translate_segments(
        segments,
        target_language,
        cache_path,
        workers,
        state,
        translate_backend=translate_backend,
        translate_model=translate_model,
    )

    background_wav = video_path.with_suffix(".background.wav")
    extract_background_audio(
        video_path, background_wav, force=fresh, dialogue_wav=dialogue_wav
    )
    state.save(stage="background_ready", background=str(background_wav))

    dubbed_wav = synthesize_dubbed_audio(
        translated,
        chunks_dir,
        workers,
        state,
        target_language,
        timeline_duration=_probe_duration(video_path),
    )
    master_final_video(
        video_path,
        output_video,
        background_wav,
        workers,
        state,
        dub_wav=dubbed_wav,
        force=fresh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dub a video using subtitle timings.")
    parser.add_argument("--input", default="movie.mp4", help="Input video file")
    parser.add_argument("--srt", default="movie.srt", help="Subtitle file (.srt)")
    parser.add_argument("--output", default="movie_dubbed.mp4", help="Output video file")
    parser.add_argument("--language", default="hi", help="Target dub language code (hi, en, es, ...)")
    parser.add_argument("--duration", type=float, help="Only dub the first N seconds")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Parallel workers for translate/TTS/render",
    )
    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Use plain Whisper pipeline (ignore SRT)",
    )
    parser.add_argument(
        "--no-srt-ranges",
        metavar="RANGES",
        help="Ranges where SRT is off and plain Whisper is used (e.g. '45-60,1:30-2:00')",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model when using --whisper-only or --no-srt-ranges",
    )
    parser.add_argument(
        "--vllm-speakers",
        action="store_true",
        help="Use a vLLM server to identify who speaks each line and summarize speech time",
    )
    parser.add_argument("--vllm-url", default=None, help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--vllm-model", default=None, help="vLLM model name")
    parser.add_argument("--vllm-speaker-cache", default=None, help="Cache file for vLLM speaker labels")
    parser.add_argument(
        "--translate-backend",
        choices=("auto", "parallel", "minimax", "cloud", "google"),
        default=os.environ.get("DUB_TRANSLATE_BACKEND", "auto"),
        help="Translation backend (default: auto — parallel Minimax + Google)",
    )
    args = parser.parse_args()

    vllm_kwargs = {
        "use_vllm_speakers": args.vllm_speakers,
        "vllm_base_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "vllm_speaker_cache": Path(args.vllm_speaker_cache) if args.vllm_speaker_cache else None,
    }

    translate_kwargs = {
        "translate_backend": args.translate_backend,
    }

    use_whisper_pipeline = args.whisper_only or bool(args.no_srt_ranges)
    if use_whisper_pipeline:
        from dub_movie_whisper import dub_with_whisper_alignment, parse_time_ranges

        if args.whisper_only and args.no_srt_ranges:
            parser.error("--whisper-only cannot be combined with --no-srt-ranges")
        if args.duration is None:
            parser.error("--duration is required with --whisper-only or --no-srt-ranges")

        dub_with_whisper_alignment(
            input_video=Path(args.input),
            subtitle_file=None if args.whisper_only else Path(args.srt),
            output_video=Path(args.output),
            target_language=args.language,
            duration=args.duration,
            workers=args.workers,
            whisper_model=args.whisper_model,
            no_srt_ranges=parse_time_ranges(args.no_srt_ranges),
            whisper_only=args.whisper_only,
            **vllm_kwargs,
            **translate_kwargs,
        )
        return

    dub_from_subtitles(
        input_video=Path(args.input),
        subtitle_file=Path(args.srt),
        output_video=Path(args.output),
        target_language=args.language,
        duration=args.duration,
        workers=args.workers,
        **vllm_kwargs,
        **translate_kwargs,
    )


if __name__ == "__main__":
    main()
