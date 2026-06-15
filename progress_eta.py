"""Progress reporting with ETA for dub pipeline phases."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# Wall-clock tuning from full-movie runs (4 workers unless noted).
SEC_PITCH_LINE = 0.28       # ~9 min for 1927 lines (probe ~5m + analysis ~4m)
SEC_PITCH_PASS_LINE = 0.16  # ~5 min pitch probe pass on 1927 lines
SEC_TRANSLATE_LINE = 0.45
SEC_TTS_BATCH = 1.15        # ~5 min for ~778 remaining lines at 4 workers
SEC_BACKGROUND_EXTRACT = 90.0
SEC_RENDER_BASE = 180.0
SEC_RENDER_PER_VIDEO_SEC = 0.035
SEC_VLLM_SPEAKER_LINE = 8.0
DEFAULT_DUB_LINE_RATIO = 0.82


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"{minutes}m {secs}s"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if minutes:
        return f"{hours}h {minutes}m"
    return f"{hours}h {secs}s"


def count_translation_cache(cache_path: Path | None) -> int:
    if not cache_path or not cache_path.exists():
        return 0
    try:
        return len(json.loads(cache_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return 0


def count_tts_chunks(chunks_dir: Path | None) -> int:
    if not chunks_dir or not chunks_dir.exists():
        return 0
    return sum(1 for path in chunks_dir.glob("*.mp3") if path.stat().st_size >= 256)


def voice_cache_entry_count(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if payload.get("version") != 1:
        return 0
    entries = payload.get("entries")
    return len(entries) if isinstance(entries, dict) else 0


def translation_cache_has_profiles(cache_path: Path | None) -> int:
    """Return count of translation-cache lines that include TTS profiles."""
    if not cache_path or not cache_path.exists():
        return 0
    try:
        cached_list = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(cached_list, list):
        return 0
    return sum(
        1
        for item in cached_list
        if isinstance(item, dict) and item.get("key") and item.get("profile")
    )


def estimate_remaining_seconds(
    *,
    srt_lines: int,
    dub_lines: int | None = None,
    translated_done: int = 0,
    tts_done: int = 0,
    video_duration_sec: float = 0.0,
    workers: int = 4,
    include_pitch: bool = True,
    include_translate: bool = True,
    include_tts: bool = True,
    include_background: bool = True,
    include_render: bool = True,
    use_vllm_speakers: bool = False,
) -> int:
    workers = max(1, workers)
    dub_lines = dub_lines or max(translated_done, int(srt_lines * DEFAULT_DUB_LINE_RATIO))
    total = 0.0

    if include_pitch:
        total += srt_lines * SEC_PITCH_LINE
    if use_vllm_speakers:
        total += srt_lines * SEC_VLLM_SPEAKER_LINE
    if include_translate:
        total += max(0, dub_lines - translated_done) * SEC_TRANSLATE_LINE
    if include_tts:
        pending = max(0, dub_lines - tts_done)
        total += (pending / workers) * SEC_TTS_BATCH
    if include_background:
        total += SEC_BACKGROUND_EXTRACT
    if include_render:
        total += SEC_RENDER_BASE + video_duration_sec * SEC_RENDER_PER_VIDEO_SEC
    return int(total)


def print_pipeline_eta(
    *,
    srt_lines: int,
    cache_path: Path | None,
    chunks_dir: Path | None,
    video_duration_sec: float,
    workers: int,
    use_vllm_speakers: bool = False,
    voice_cache_path: Path | None = None,
) -> None:
    translated = count_translation_cache(cache_path)
    tts_done = count_tts_chunks(chunks_dir)
    dub_lines = max(translated, int(srt_lines * DEFAULT_DUB_LINE_RATIO))
    voice_cached = voice_cache_entry_count(voice_cache_path) >= srt_lines
    profile_lines = translation_cache_has_profiles(cache_path)
    pitch_cached = voice_cached or profile_lines >= int(dub_lines * 0.95)
    remaining = estimate_remaining_seconds(
        srt_lines=srt_lines,
        dub_lines=dub_lines,
        translated_done=translated,
        tts_done=tts_done,
        video_duration_sec=video_duration_sec,
        workers=workers,
        include_pitch=not pitch_cached,
        use_vllm_speakers=use_vllm_speakers and not pitch_cached,
    )
    parts: list[str] = []
    if pitch_cached:
        parts.append("pitch cached")
    if translated:
        parts.append(f"translation {translated} cached")
    if tts_done:
        parts.append(f"TTS {tts_done} cached")
    cache_note = f" ({', '.join(parts)})" if parts else ""
    print(
        f"Estimated time to finish: ~{format_eta(remaining)} "
        f"[{srt_lines} SRT lines, ~{dub_lines} dub lines{cache_note}]"
    )


class PhaseTimer:
    """Track progress within one pipeline phase and print ETA periodically."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        initial: int = 0,
        report_every: int | None = None,
        fallback_rate: float | None = None,
    ) -> None:
        self.label = label
        self.total = max(total, 1)
        self.initial = initial
        self.done = initial
        self.report_every = report_every or max(25, self.total // 20)
        self.fallback_rate = fallback_rate or (1.0 / SEC_PITCH_LINE)
        self.start = time.monotonic()
        self._last_report = initial
        self._lock = threading.Lock()

    def _remaining_seconds(self) -> float:
        remaining = max(0, self.total - self.done)
        elapsed = time.monotonic() - self.start
        processed = self.done - self.initial
        if processed > 0 and elapsed > 0:
            return remaining * (elapsed / processed)
        return remaining / self.fallback_rate

    def announce(self) -> None:
        remaining_count = max(0, self.total - self.done)
        print(
            f"{self.label}: {self.done}/{self.total} — "
            f"~{remaining_count} remaining, ETA ~{format_eta(self._remaining_seconds())}"
        )

    def tick(self, step: int = 1) -> None:
        with self._lock:
            self.done += step
            if (
                self.done >= self.total
                or self.done - self.initial == 1
                or self.done - self._last_report >= self.report_every
            ):
                self._last_report = self.done
                self._print_progress()

    def _print_progress(self) -> None:
        remaining_count = max(0, self.total - self.done)
        elapsed = time.monotonic() - self.start
        print(
            f"{self.label}: {self.done}/{self.total} — "
            f"~{remaining_count} left, ETA ~{format_eta(self._remaining_seconds())} "
            f"(elapsed {format_eta(elapsed)})"
        )

    def finish(self) -> None:
        elapsed = time.monotonic() - self.start
        print(f"{self.label} complete: {self.done}/{self.total} in {format_eta(elapsed)}")
