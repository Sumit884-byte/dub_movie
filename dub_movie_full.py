"""Dub the full movie (~105 min) with v8 pipeline settings."""

from pathlib import Path

from dub_movie import default_workers
from dub_movie_whisper import dub_with_whisper_alignment

OUTPUT = Path("movie_dubbed_full_v8.mp4")

if __name__ == "__main__":
    dub_with_whisper_alignment(
        input_video=Path("movie.mp4"),
        subtitle_file=Path("movie.srt"),
        output_video=OUTPUT,
        target_language="hi",
        duration=None,
        cache_path=Path("translated_segments_full_v8.json"),
        chunks_dir=Path("temp_chunks_full_v8"),
        workers=default_workers(),
        state_path=Path("movie_dubbed_full_v8.state.json"),
        fresh=False,
        srt_timing_only=True,
        translate_backend="parallel",
        # Heuristic + SRT speaker tags only — vLLM is too slow for ~1900 lines.
        use_vllm_speakers=False,
    )
