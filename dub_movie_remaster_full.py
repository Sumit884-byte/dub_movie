"""Remux full movie from existing dub timeline + background (fix audio mix)."""

from pathlib import Path

from dub_movie import JobState, default_workers, master_final_video

OUTPUT = Path("movie_dubbed_full_v8.mp4")

if __name__ == "__main__":
    master_final_video(
        input_video=Path("movie.mp4"),
        output_video=OUTPUT,
        background_wav=Path("movie.background.wav"),
        workers=default_workers(),
        state=JobState(Path("movie_dubbed_full_v8.state.json")),
        dub_wav=Path("temp_chunks_full_v8/dub_timeline.wav"),
        force=True,
    )
