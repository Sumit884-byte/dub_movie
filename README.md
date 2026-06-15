# dub_movie

Automated movie dubbing pipeline: Japanese source video + SRT subtitles → translated speech (Edge TTS), pitch-matched per character, mixed over a vocal-free background bed.

Designed for anime-style content with named cast profiles (Doraemon cast, etc.), on-screen subtitle sync, and resumable long runs.

## Requirements

- **Python 3.10+**
- **ffmpeg** on `PATH`
- GPU optional (Whisper runs on CPU by default; local NLLB translation benefits from CUDA)

### Python packages

Core:

```bash
pip install edge-tts moviepy numpy librosa httpx deep-translator faster-whisper
```

Optional:

| Package | Used for |
|---------|----------|
| `torch`, `transformers` | Local NLLB translation (`local_translate.py`) |
| `demucs` | Higher-quality vocal separation (`vocal_separation.py`) |

## Input files

Place in the working directory (or pass paths via CLI):

| File | Purpose |
|------|---------|
| `movie.mp4` | Source video (default input) |
| `movie.srt` | Japanese subtitles with optional speaker tags `（キャラ名）` |

## Quick start

Dub the first 60 seconds into Hindi (Whisper-aligned timing):

```bash
python dub_movie_whisper.py --input movie.mp4 --srt movie.srt --output movie_dubbed_1min.mp4 --duration 60 --language hi
```

Exact SRT timing (recommended when subtitles must match on-screen text):

```bash
python dub_movie_whisper.py --srt-timing-only --duration 60 --language hi
```

SRT-only pipeline (no Whisper alignment):

```bash
python dub_movie.py --input movie.mp4 --srt movie.srt --output movie_dubbed.mp4 --language hi --duration 300
```

## Entry points

### `dub_movie_whisper.py` (recommended)

Full pipeline with Whisper speech detection, SRT alignment, pitch analysis, and character-aware translation.

```bash
python dub_movie_whisper.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `movie.mp4` | Source video |
| `--srt` | `movie.srt` | Subtitle file (optional with `--whisper-only`) |
| `--output` | `movie_dubbed_1min.mp4` | Output video |
| `--language` | `hi` | Target language code (`hi`, `en`, `es`, …) |
| `--duration` | `60` | Process only the first N seconds |
| `--whisper-model` | `base` | `tiny` / `base` / `small` / `medium` |
| `--srt-timing-only` | off | Use exact SRT timestamps (no Whisper shift) |
| `--whisper-only` | off | Ignore SRT; dub from Whisper transcription |
| `--no-srt-ranges` | — | Ranges where SRT is disabled, e.g. `45-60,1:30-2:00` |
| `--vllm-speakers` | off | LLM speaker identification per line |
| `--vllm-url` | `$VLLM_BASE_URL` | OpenAI-compatible API base URL |
| `--vllm-model` | first served model | Model name |
| `--vllm-speaker-cache` | — | Cache file for speaker labels |
| `--translate-backend` | `auto` | `auto`, `parallel`, `minimax`, `cloud`, `google` |
| `--workers` | 2–8 (CPU-based) | Parallel translate / TTS / render workers |
| `--fresh` | off | Regenerate clip, background, and final render |

### `dub_movie.py`

Unified CLI: SRT-timing dub by default; delegates to the Whisper pipeline when `--whisper-only` or `--no-srt-ranges` is set (requires `--duration`).

```bash
python dub_movie.py --input movie.mp4 --srt movie.srt --output movie_dubbed.mp4 --language hi
```

## Preset runners

Scripts with baked-in paths and cache names for iterative development:

| Script | Duration | Notes |
|--------|----------|-------|
| `dub_movie_1min.py` | 60 s | `--srt-timing-only`, `fresh=True` |
| `dub_movie_5min.py` | 300 s | Parallel translate + vLLM speakers (Ollama) |
| `dub_movie_10min.py` | 600 s | `--srt-timing-only` |
| `dub_movie_full.py` | full movie | No vLLM speakers (~1900 lines); parallel translate |
| `dub_movie_clip.py` | 300 s | Legacy SRT-only path (`dub_from_subtitles`) |
| `dub_movie_remaster_full.py` | — | Re-mux only from existing timeline + background |
| `dub_combine_timeline_full.py` | — | Combine TTS MP3 chunks into `dub_timeline.wav` |

Run from the repo root with `movie.mp4` and `movie.srt` present:

```bash
python dub_movie_1min.py
```

## Pipeline stages

Jobs write progress to a `.state.json` file and can resume after interruption.

1. **Prepare clip** — trim video to `--duration` if set  
2. **Dialogue extract** — mono dialogue channel for pitch analysis  
3. **Whisper** (unless `--srt-timing-only`) — speech timing; align to SRT  
4. **Speaker ID** (optional) — vLLM labels who speaks each line  
5. **Voice analysis** — F0 pitch per segment; map to Edge TTS rate/pitch  
6. **Translate** — context-aware JA→target via character profiles  
7. **Background** — vocal-reduced bed (center-channel / 5.1 routing; optional Demucs)  
8. **TTS** — Edge TTS per line, cached as MP3 chunks  
9. **Render** — mux dubbed timeline + background into output MP4  

State stages (in order): `start` → `whisper_audio_ready` → `whisper_done` → `aligned` → `speakers_vllm` → `voice_analyzed` → `translating` → `translated` → `background_ready` → `synthesizing` → `synthesized` → `rendering` → `done`.

## Translation backends

Controlled by `--translate-backend` or `DUB_TRANSLATE_BACKEND`.

| Backend | Behavior |
|---------|----------|
| `auto` | Parallel Minimax + Google when both available |
| `parallel` | Shard lines across configured backends |
| `minimax` / `cloud` | Ollama/OpenAI-compatible chat API (see env vars) |
| `google` | `deep-translator` Google Translate |

### Translation environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DUB_TRANSLATE_BACKEND` | `auto` | Default backend |
| `DUB_TRANSLATE_CLOUD_URL` | `http://localhost:11434/v1` | Cloud/Ollama API URL |
| `DUB_TRANSLATE_CLOUD_MODEL` | `minimax-m2.5:cloud` | Chat model for batch translate |
| `DUB_TRANSLATE_BATCH` | `10` | Lines per cloud batch |
| `DUB_TRANSLATE_WORKERS` | `16` | Parallel translate threads |
| `DUB_TRANSLATE_TIMEOUT` | `120` | HTTP timeout (seconds) |
| `DUB_TRANSLATE_MAX_RETRIES` | `4` | Retry count on rate limits |
| `DUB_TRANSLATE_RETRY_BASE_SEC` | `2.0` | Exponential backoff base |
| `DUB_TRANSLATE_PARALLEL_BACKENDS` | `minimax,google` | Comma-separated shard backends |

Local NLLB (offline): set `DUB_TRANSLATE_MODEL` (default `facebook/nllb-200-distilled-600M`) and `DUB_TRANSLATE_DEVICE` (`cuda` / `cpu`).

## vLLM speaker identification

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible API |
| `VLLM_MODEL` | `""` | Model name (empty = first served) |
| `VLLM_API_KEY` | `EMPTY` | API key |
| `VLLM_SPEAKER_BATCH` | `8` | Lines per batch |
| `VLLM_TIMEOUT` | `600` | Request timeout (seconds) |

## Caches and resume

The pipeline writes artifacts next to the output path so reruns skip finished work:

| Artifact | Purpose |
|----------|---------|
| `*.state.json` | Stage checkpoint / resume |
| `translated_segments_*.json` | Translation cache |
| `*.voice.json` | Pitch / TTS profile cache |
| `*.speakers.json` | vLLM speaker label cache |
| `temp_chunks_*/` | Per-line Edge TTS MP3 files |
| `*.background.wav` | Vocal-reduced background bed |

Use `--fresh` to force regeneration of clip, background, and final render. Delete specific cache files to redo individual stages.

## Supporting modules

| Module | Role |
|--------|------|
| `character_context.py` | Cast profiles, tone detection, contextual translation, TTS voice mapping |
| `fast_translate.py` | Multi-backend translation with parallel sharding |
| `local_translate.py` | Offline NLLB-200 translation |
| `voice_analysis.py` | F0 pitch estimation (librosa) and TTS profile tuning |
| `speaker_id_vllm.py` | LLM-based speaker assignment |
| `vocal_separation.py` | Demucs two-stem vocal/instrumental split |
| `progress_eta.py` | Phase timing and ETA estimates |

## Mix levels

Final audio uses a full background bed with dub slightly above it (`BACKGROUND_MIX_GAIN=1.5`, `DUB_MIX_GAIN=1.25` in `dub_movie.py`).
