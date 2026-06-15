from pathlib import Path

from dub_movie import default_workers
from dub_movie_whisper import dub_with_whisper_alignment

OUTPUT = Path("movie_dubbed_5min_v8.mp4")
VLLM_URL = "http://localhost:11434/v1"  # Ollama OpenAI-compatible API
VLLM_MODEL = "llama3.2:1b"

if __name__ == "__main__":
    dub_with_whisper_alignment(
        input_video=Path("movie.mp4"),
        subtitle_file=Path("movie.srt"),
        output_video=OUTPUT,
        target_language="hi",
        duration=300,
        cache_path=Path("translated_segments_5min_v8.json"),
        chunks_dir=Path("temp_chunks_5min_v8"),
        workers=default_workers(),
        state_path=Path("movie_dubbed_5min_v8.state.json"),
        fresh=False,
        srt_timing_only=True,
        translate_backend="parallel",
        use_vllm_speakers=True,
        vllm_base_url=VLLM_URL,
        vllm_model=VLLM_MODEL,
        vllm_speaker_cache=Path("speaker_labels_5min_v8.json"),
    )
