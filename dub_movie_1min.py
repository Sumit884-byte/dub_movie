from pathlib import Path

from dub_movie import default_workers
from dub_movie_whisper import dub_with_whisper_alignment

OUTPUT = Path("movie_dubbed_1min_v7.mp4")

if __name__ == "__main__":
    dub_with_whisper_alignment(
        input_video=Path("movie.mp4"),
        subtitle_file=Path("movie.srt"),
        output_video=OUTPUT,
        target_language="hi",
        duration=60,
        cache_path=Path("translated_segments_1min_v7.json"),
        chunks_dir=Path("temp_chunks_1min_v7"),
        workers=default_workers(),
        state_path=Path("movie_dubbed_1min_v7.state.json"),
        fresh=True,
        srt_timing_only=True,
    )
