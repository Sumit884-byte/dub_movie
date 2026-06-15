"""Combine all TTS MP3 chunks into one dub timeline WAV (no re-TTS, no mux)."""

import json
from pathlib import Path

from dub_movie import JobState, _probe_duration, build_dub_timeline_wav

SEGMENTS = Path("translated_segments_full_v8.json")
CHUNKS = Path("temp_chunks_full_v8")
OUTPUT = CHUNKS / "dub_timeline.wav"
VIDEO = Path("movie.mp4")
STATE = Path("movie_dubbed_full_v8.state.json")

if __name__ == "__main__":
    segments = json.load(open(SEGMENTS))
    duration = _probe_duration(VIDEO)
    used, skipped = build_dub_timeline_wav(segments, CHUNKS, OUTPUT, duration)
    state = JobState(STATE)
    state.save(stage="synthesized", synthesized=used, total=len(segments))
    print(f"Combined {used} chunks -> {OUTPUT.resolve()} ({skipped} skipped)")
