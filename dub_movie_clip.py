from pathlib import Path

from dub_movie import dub_from_subtitles, default_workers

if __name__ == "__main__":
    dub_from_subtitles(
        input_video=Path("movie.mp4"),
        subtitle_file=Path("movie.srt"),
        output_video=Path("movie_dubbed_5min.mp4"),
        target_language="hi",
        duration=300,
        cache_path=Path("translated_segments_5min_v2.json"),
        chunks_dir=Path("temp_chunks_5min_v2"),
        workers=default_workers(),
        state_path=Path("movie_dubbed_5min.state.json"),
    )
