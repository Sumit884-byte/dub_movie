"""Align SRT dialogue to speech; sync on-screen top subtitles; match voice pitch."""

import argparse
import os
import subprocess
from pathlib import Path

from dub_movie import (
    JobState,
    default_workers,
    extract_background_audio,
    master_final_video,
    parse_srt,
    segment_key,
    synthesize_dubbed_audio,
    translate_segments,
    prepare_input_video,
    voice_profile_for_speaker,
    _probe_duration,
)
from character_context import propagate_speaker
from progress_eta import print_pipeline_eta
from voice_analysis import (
    analyze_voices_for_segments,
    attach_pitch_to_segments,
    extract_dialogue_mono,
    load_voiced_segments,
    save_voice_cache,
)
from speaker_id_vllm import identify_speakers_vllm


def _parse_timestamp(token: str) -> float:
    token = token.strip()
    if ":" in token:
        parts = token.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        raise ValueError(f"Invalid timestamp: {token}")
    return float(token)


def parse_time_ranges(spec: str | None) -> list[tuple[float, float]]:
    """Parse ranges like '45-60,1:30-2:00,0:45:30-0:46:00'."""
    if not spec or not spec.strip():
        return []

    ranges: list[tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid range (expected start-end): {part}")
        start_text, end_text = part.split("-", 1)
        start = _parse_timestamp(start_text)
        end = _parse_timestamp(end_text)
        if end <= start:
            raise ValueError(f"Range end must be after start: {part}")
        ranges.append((start, end))
    return sorted(ranges, key=lambda item: item[0])


def _segment_in_ranges(seg: dict, ranges: list[tuple[float, float]]) -> bool:
    if not ranges:
        return False
    mid = (seg["start"] + seg["end"]) / 2
    return any(start <= mid <= end for start, end in ranges)


def whisper_to_dub_segments(whisper_segments: list[dict]) -> list[dict]:
    """Convert raw Whisper speech spans into dubbing segment dicts."""
    segments = []
    for whisper in whisper_segments:
        text = whisper["text"].strip()
        if not text:
            continue
        start = float(whisper["start"])
        end = float(whisper["end"])
        item = {
            "start": start,
            "end": end,
            "srt_start": start,
            "srt_end": end,
            "text": text,
            "source_ja": text,
            "speaker": None,
            "on_screen": False,
            "position": None,
            "timing_source": "whisper_only",
        }
        item["key"] = segment_key(item)
        segments.append(item)
    return segments


def merge_srt_and_whisper_segments(
    srt_aligned: list[dict],
    whisper_segments: list[dict],
    no_srt_ranges: list[tuple[float, float]],
) -> list[dict]:
    """Use SRT everywhere except the given ranges, where plain Whisper segments are used."""
    if not no_srt_ranges:
        return srt_aligned

    srt_part = [seg for seg in srt_aligned if not _segment_in_ranges(seg, no_srt_ranges)]
    whisper_part = [
        seg
        for seg in whisper_to_dub_segments(whisper_segments)
        if _segment_in_ranges(seg, no_srt_ranges)
    ]
    merged = _prevent_segment_overlaps(sorted(srt_part + whisper_part, key=lambda seg: seg["start"]))

    print(
        f"Mixed pipeline: {len(srt_part)} SRT segments, "
        f"{len(whisper_part)} Whisper-only segments "
        f"across {len(no_srt_ranges)} disabled-SRT range(s)."
    )
    for start, end in no_srt_ranges:
        print(f"  SRT off: {start:.1f}s - {end:.1f}s")
    return merged


def extract_audio_wav(video_path: Path, wav_path: Path, duration: float | None = None) -> Path:
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000"]
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.append(str(wav_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def transcribe_speech(wav_path: Path, duration: float, model_size: str = "base") -> list[dict]:
    from faster_whisper import WhisperModel

    print(f"Running Whisper ({model_size}) to detect speech timing...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(wav_path),
        language="ja",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    speech = []
    for seg in segments:
        if seg.start >= duration:
            break
        if not seg.text.strip():
            continue
        speech.append(
            {
                "start": float(seg.start),
                "end": float(min(seg.end, duration)),
                "text": seg.text.strip(),
            }
        )

    print(f"Whisper found {len(speech)} speech segments.")
    return speech


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


MAX_TIMING_NUDGE = 0.35
MAX_TIMING_EXTRA = 0.4
MAX_TIMING_RATIO = 1.35


def _timing_from_whisper_match(srt: dict, whisper: dict) -> tuple[float, float]:
    """Keep the SRT dub slot; nudge start/end slightly when Whisper agrees."""
    srt_start, srt_end = srt["srt_start"], srt["srt_end"]
    srt_len = max(srt_end - srt_start, 0.1)
    w_start, w_end = whisper["start"], whisper["end"]

    start, end = srt_start, srt_end

    if w_start < srt_start and srt_start - w_start <= MAX_TIMING_NUDGE:
        start = w_start
    elif w_start > srt_start and w_start - srt_start <= MAX_TIMING_NUDGE:
        start = w_start

    if w_end > srt_end and w_end - srt_end <= MAX_TIMING_EXTRA:
        end = w_end

    start = max(start, srt_start - MAX_TIMING_NUDGE)
    end = min(end, srt_end + MAX_TIMING_EXTRA, start + srt_len * MAX_TIMING_RATIO)
    return start, max(end, start + 0.1)


def _prevent_segment_overlaps(segments: list[dict]) -> list[dict]:
    if not segments:
        return segments

    ordered = sorted(segments, key=lambda s: s["start"])
    for index in range(1, len(ordered)):
        prev = ordered[index - 1]
        cur = ordered[index]
        if cur["start"] >= prev["end"]:
            continue
        split = (prev["end"] + cur["start"]) / 2
        prev["end"] = max(prev["start"] + 0.1, split)
        cur["start"] = prev["end"]
        cur["key"] = segment_key(cur)
    return ordered


def align_srt_to_whisper(srt_segments: list[dict], whisper_segments: list[dict]) -> list[dict]:
    """
    On-screen top subtitles ({\\an7-9}) use exact SRT timing so dub matches visible text.
    Other lines use Whisper speech detection when possible.
    """
    aligned: list[dict] = []
    on_screen_count = 0
    whisper_count = 0
    srt_fallback = 0

    whisper_pool = list(enumerate(whisper_segments))
    used_whisper: set[int] = set()

    for srt in srt_segments:
        if srt.get("on_screen"):
            item = {
                **srt,
                "start": srt["srt_start"],
                "end": srt["srt_end"],
                "timing_source": "on_screen_srt",
            }
            item["key"] = segment_key(item)
            aligned.append(item)
            on_screen_count += 1
            continue

        best_idx = None
        best_score = 0.0
        srt_len = max(srt["srt_end"] - srt["srt_start"], 0.1)

        for idx, w in whisper_pool:
            if idx in used_whisper:
                continue
            overlap = _overlap(srt["srt_start"], srt["srt_end"], w["start"], w["end"])
            if overlap <= 0:
                srt_mid = (srt["srt_start"] + srt["srt_end"]) / 2
                w_mid = (w["start"] + w["end"]) / 2
                if abs(srt_mid - w_mid) > 4.0:
                    continue
                score = 0.15
            else:
                w_len = max(w["end"] - w["start"], 0.1)
                score = overlap / min(srt_len, w_len)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.15 and whisper_segments:
            w = whisper_segments[best_idx]
            used_whisper.add(best_idx)
            start, end = _timing_from_whisper_match(srt, w)
            item = {
                **srt,
                "start": start,
                "end": end,
                "whisper_text": w["text"],
                "timing_source": "whisper",
            }
            whisper_count += 1
        else:
            item = {
                **srt,
                "start": srt["srt_start"],
                "end": srt["srt_end"],
                "timing_source": "srt",
            }
            srt_fallback += 1

        item["key"] = segment_key(item)
        aligned.append(item)

    aligned = _prevent_segment_overlaps(aligned)

    print(
        f"Timing: {on_screen_count} on-screen (top subtitle sync), "
        f"{whisper_count} Whisper-aligned, {srt_fallback} SRT fallback."
    )
    return aligned


def dub_with_whisper_alignment(
    input_video: Path,
    subtitle_file: Path | None,
    output_video: Path,
    target_language: str = "hi",
    duration: float = 60,
    cache_path: Path | None = None,
    chunks_dir: Path | None = None,
    workers: int | None = None,
    state_path: Path | None = None,
    whisper_model: str = "base",
    no_srt_ranges: list[tuple[float, float]] | None = None,
    whisper_only: bool = False,
    use_vllm_speakers: bool = False,
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_speaker_cache: Path | None = None,
    fresh: bool = False,
    srt_timing_only: bool = False,
    translate_backend: str = "auto",
    translate_model: str | None = None,
    voice_cache_path: Path | None = None,
) -> None:
    workers = workers or default_workers()
    cache_path = cache_path or output_video.with_suffix(".segments.json")
    chunks_dir = chunks_dir or Path("temp_chunks") / output_video.stem
    state_path = state_path or output_video.with_suffix(".state.json")
    state = JobState(state_path)
    no_srt_ranges = no_srt_ranges or []

    if whisper_only:
        print("Pipeline: Whisper-only (SRT disabled)")
    elif srt_timing_only:
        print("Pipeline: SRT timing only (exact subtitle sync, vocal-free bed + dub)")
    elif no_srt_ranges:
        print(f"Pipeline: mixed SRT + Whisper-only in {len(no_srt_ranges)} range(s)")
    else:
        print("Pipeline: SRT with Whisper timing alignment")

    print(f"Workers: {workers} | Duration: {duration}s | Whisper model: {whisper_model}")
    print(f"Translation: {translate_backend}")

    video_path = prepare_input_video(input_video, duration, force=fresh)
    dialogue_wav = video_path.with_suffix(".dialogue.wav")
    extract_dialogue_mono(video_path, dialogue_wav, duration=duration, force=fresh)

    if whisper_only:
        whisper_wav = video_path.with_suffix(".whisper.wav")
        extract_audio_wav(video_path, whisper_wav, duration=duration)
        whisper_segments = transcribe_speech(whisper_wav, duration, model_size=whisper_model)
        segments = propagate_speaker(_prevent_segment_overlaps(whisper_to_dub_segments(whisper_segments)))
        print(f"Whisper-only: {len(segments)} speech segments.")
    elif srt_timing_only:
        if subtitle_file is None:
            raise RuntimeError("Subtitle file is required for SRT timing mode.")
        srt_segments = parse_srt(subtitle_file, max_duration=duration)
        if not srt_segments:
            raise RuntimeError(f"No usable subtitle lines in first {duration}s of {subtitle_file}")
        segments = propagate_speaker(srt_segments)
        print(f"Parsed {len(segments)} SRT lines (exact subtitle timing).")
    else:
        whisper_wav = video_path.with_suffix(".whisper.wav")
        extract_audio_wav(video_path, whisper_wav, duration=duration)
        state.save(stage="whisper_audio_ready")
        whisper_segments = transcribe_speech(whisper_wav, duration, model_size=whisper_model)
        state.save(stage="whisper_done", whisper_segments=len(whisper_segments))

        if subtitle_file is None:
            raise RuntimeError("Subtitle file is required unless --whisper-only is set.")
        srt_segments = parse_srt(subtitle_file, max_duration=duration)
        if not srt_segments:
            raise RuntimeError(f"No usable subtitle lines in first {duration}s of {subtitle_file}")

        on_screen = sum(1 for seg in srt_segments if seg.get("on_screen"))
        print(f"Parsed {len(srt_segments)} SRT lines ({on_screen} top on-screen subtitles).")

        aligned = propagate_speaker(align_srt_to_whisper(srt_segments, whisper_segments))
        segments = merge_srt_and_whisper_segments(aligned, whisper_segments, no_srt_ranges)

    state.save(stage="aligned", aligned=len(segments))

    voice_cache_path = voice_cache_path or output_video.with_suffix(".voice.json")
    if fresh and voice_cache_path.exists():
        voice_cache_path.unlink()

    print_pipeline_eta(
        srt_lines=len(segments),
        cache_path=cache_path,
        chunks_dir=chunks_dir,
        video_duration_sec=_probe_duration(video_path),
        workers=workers,
        use_vllm_speakers=use_vllm_speakers,
        voice_cache_path=voice_cache_path,
    )

    voiced = load_voiced_segments(
        segments,
        voice_cache_path=voice_cache_path,
        translation_cache_path=cache_path,
    )
    if voiced is None:
        segments = attach_pitch_to_segments(segments, dialogue_wav)

        speaker_summary: dict[str, dict] = {}
        if use_vllm_speakers:
            speaker_cache = vllm_speaker_cache or output_video.with_suffix(".speakers.json")
            segments, speaker_summary = identify_speakers_vllm(
                segments,
                cache_path=speaker_cache,
                base_url=vllm_base_url,
                model=vllm_model,
            )
            state.save(stage="speakers_vllm", speaker_summary=speaker_summary)

        voiced = analyze_voices_for_segments(
            segments, dialogue_wav, voice_profile_for_speaker, target_language
        )
        save_voice_cache(voice_cache_path, voiced)
        print(f"Saved voice/pitch cache: {voice_cache_path.name}")

    state.save(stage="voice_analyzed", lines=len(voiced))

    translated = translate_segments(
        voiced,
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
    parser = argparse.ArgumentParser(
        description="Dub video: SRT text, on-screen subtitle sync, Whisper timing, pitch-matched voices."
    )
    parser.add_argument("--input", default="movie.mp4")
    parser.add_argument("--srt", default="movie.srt", help="Subtitle file (.srt); optional with --whisper-only")
    parser.add_argument("--output", default="movie_dubbed_1min.mp4")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--whisper-model", default="base", help="tiny/base/small/medium")
    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Ignore SRT entirely; transcribe and dub from Whisper speech segments",
    )
    parser.add_argument(
        "--no-srt-ranges",
        metavar="RANGES",
        help="Comma-separated ranges where SRT is disabled and plain Whisper is used "
        "(e.g. '45-60,1:30-2:00')",
    )
    parser.add_argument(
        "--srt-timing-only",
        action="store_true",
        help="Use exact SRT subtitle times (no Whisper timing shifts); recommended for sync",
    )
    parser.add_argument(
        "--vllm-speakers",
        action="store_true",
        help="Use a vLLM server to identify who speaks each line and summarize speech time",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate clip, vocal-free background, and final render from scratch",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="vLLM OpenAI-compatible base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    )
    parser.add_argument("--vllm-model", default=None, help="vLLM model name (default: first served model)")
    parser.add_argument(
        "--vllm-speaker-cache",
        default=None,
        help="Cache file for vLLM speaker assignments",
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--translate-backend",
        choices=("auto", "parallel", "minimax", "cloud", "google"),
        default=os.environ.get("DUB_TRANSLATE_BACKEND", "auto"),
        help="Translation backend (default: auto — parallel Minimax + Google)",
    )
    args = parser.parse_args()

    subtitle_file = None if args.whisper_only else Path(args.srt)
    no_srt_ranges = parse_time_ranges(args.no_srt_ranges)
    if args.whisper_only and no_srt_ranges:
        parser.error("--whisper-only cannot be combined with --no-srt-ranges")

    dub_with_whisper_alignment(
        input_video=Path(args.input),
        subtitle_file=subtitle_file,
        output_video=Path(args.output),
        target_language=args.language,
        duration=args.duration,
        workers=args.workers,
        whisper_model=args.whisper_model,
        no_srt_ranges=no_srt_ranges,
        whisper_only=args.whisper_only,
        use_vllm_speakers=args.vllm_speakers,
        vllm_base_url=args.vllm_url,
        vllm_model=args.vllm_model,
        vllm_speaker_cache=Path(args.vllm_speaker_cache) if args.vllm_speaker_cache else None,
        fresh=args.fresh,
        srt_timing_only=args.srt_timing_only,
        translate_backend=args.translate_backend,
        translate_model=args.translate_model,
    )


if __name__ == "__main__":
    main()
