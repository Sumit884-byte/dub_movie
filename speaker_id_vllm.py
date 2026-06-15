"""Identify who is speaking, when, and how much using a vLLM OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx

from character_context import CHARACTER_PROFILES, propagate_speaker

DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_MODEL = os.environ.get("VLLM_MODEL", "")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_BATCH_SIZE = int(os.environ.get("VLLM_SPEAKER_BATCH", "8"))
DEFAULT_VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "600"))

SPEAKER_ALIASES: dict[str, str] = {
    "クレア": "クレア",
    "claire": "クレア",
    "マイロ": "マイロ",
    "milo": "マイロ",
    "のび太": "のび太",
    "nobita": "のび太",
    "しずか": "しずか",
    "shizuka": "しずか",
    "ジャイアン": "ジャイアン",
    "gian": "ジャイアン",
    "スネ夫": "スネ夫",
    "suneo": "スネ夫",
    "ドラえもん": "ドラえもん",
    "doraemon": "ドラえもん",
    "unknown": "unknown",
    "": "unknown",
}


def vllm_config(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    return {
        "base_url": (base_url or DEFAULT_VLLM_BASE_URL).rstrip("/"),
        "model": model or DEFAULT_VLLM_MODEL,
        "api_key": api_key or DEFAULT_VLLM_API_KEY,
    }


def _normalize_speaker_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = name.strip()
    if not cleaned or cleaned.lower() == "unknown":
        return None
    return SPEAKER_ALIASES.get(cleaned, SPEAKER_ALIASES.get(cleaned.lower(), cleaned))


def build_cast_context() -> str:
    lines = ["Known cast (use exact Japanese speaker names in JSON):"]
    for name, profile in CHARACTER_PROFILES.items():
        if "role" not in profile:
            continue
        bits = [f"- {name}: {profile['role']}"]
        if profile.get("speaking_style"):
            bits.append(f"  style: {profile['speaking_style']}")
        if profile.get("key_characteristics"):
            bits.append(f"  voice: {profile['key_characteristics']}")
        lines.extend(bits)
    lines.extend(
        [
            "",
            "Heuristics:",
            "- のじゃ / playful coaxing -> クレア",
            "- うん… / hesitant soft boy -> マイロ",
            "- panic / help / clumsy pleading -> のび太",
            "- polite gentle girl -> しずか",
            "- loud rough bully -> ジャイアン",
            "- boastful nasal show-off -> スネ夫",
        ]
    )
    return "\n".join(lines)


def _segment_payload(index: int, seg: dict) -> dict:
    start = float(seg["start"])
    end = float(seg["end"])
    text = seg.get("source_ja") or seg.get("text") or ""
    payload = {
        "index": index,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(max(end - start, 0.0), 3),
        "text": text,
        "srt_speaker_hint": seg.get("speaker"),
        "timing_source": seg.get("timing_source"),
        "on_screen": bool(seg.get("on_screen")),
    }
    if seg.get("median_f0_hz") is not None:
        payload["median_pitch_hz"] = seg["median_f0_hz"]
    if seg.get("profile", {}).get("voice_gender"):
        payload["pitch_gender_hint"] = seg["profile"]["voice_gender"]
    return payload


def _build_prompt(batch: list[dict], cast_context: str) -> str:
    return (
        "You are annotating anime dialogue for dubbing.\n"
        "For each segment, decide WHO is speaking, using timing, subtitle hints, "
        "dialogue text, and pitch hints.\n"
        "Return ONLY valid JSON with this shape:\n"
        "{\n"
        '  "segments": [\n'
        '    {"index": 0, "speaker": "クレア", "confidence": 0.92, "reason": "short note"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- speaker must be one of the known cast names exactly as listed, or \"unknown\"\n"
        "- confidence is 0.0 to 1.0\n"
        "- respect srt_speaker_hint when confidence would otherwise be similar\n"
        "- use on_screen lines as strong evidence for the visible speaker\n"
        "- consider who spoke just before/after based on start_sec/end_sec ordering\n\n"
        f"{cast_context}\n\n"
        f"Segments JSON:\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def vllm_chat(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float = DEFAULT_VLLM_TIMEOUT,
) -> str:
    if not model:
        models_url = f"{base_url}/models"
        response = httpx.get(models_url, timeout=timeout)
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            raise RuntimeError(f"No models served by vLLM at {models_url}")
        model = models[0]["id"]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You label anime dialogue segments with speakers. Reply with JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _load_cache(cache_path: Path) -> dict[int, dict] | None:
    if not cache_path.exists():
        return None
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assignments = raw.get("assignments", raw)
    return {int(item["index"]): item for item in assignments}


def _save_cache(cache_path: Path, assignments: list[dict], summary: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "assignments": assignments,
        "speaker_summary": summary,
    }
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)


def speaker_speech_summary(segments: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    total_seconds = 0.0

    for seg in segments:
        speaker = _normalize_speaker_name(seg.get("speaker")) or "unknown"
        duration = max(float(seg["end"]) - float(seg["start"]), 0.0)
        bucket = stats.setdefault(
            speaker,
            {"lines": 0, "seconds": 0.0, "percent": 0.0},
        )
        bucket["lines"] += 1
        bucket["seconds"] = round(bucket["seconds"] + duration, 3)
        total_seconds += duration

    for bucket in stats.values():
        bucket["seconds"] = round(bucket["seconds"], 3)
        bucket["percent"] = round((bucket["seconds"] / total_seconds) * 100, 1) if total_seconds else 0.0

    return dict(sorted(stats.items(), key=lambda item: item[1]["seconds"], reverse=True))


def print_speaker_summary(summary: dict[str, dict]) -> None:
    if not summary:
        print("Speaker summary: no segments.")
        return

    print("Speaker time summary:")
    for speaker, data in summary.items():
        print(
            f"  {speaker}: {data['lines']} lines, "
            f"{data['seconds']:.1f}s ({data['percent']:.1f}%)"
        )


def _apply_assignments(segments: list[dict], assignments: dict[int, dict]) -> list[dict]:
    updated = []
    for index, seg in enumerate(segments):
        item = dict(seg)
        assignment = assignments.get(index)
        if not assignment:
            updated.append(item)
            continue

        speaker = _normalize_speaker_name(assignment.get("speaker"))
        confidence = float(assignment.get("confidence", 0.0))
        srt_hint = _normalize_speaker_name(item.get("speaker"))

        if speaker and confidence >= 0.45:
            item["speaker"] = speaker
            item["speaker_source"] = "vllm"
            item["speaker_confidence"] = round(confidence, 3)
            if assignment.get("reason"):
                item["speaker_reason"] = assignment["reason"]
        elif srt_hint:
            item["speaker"] = srt_hint
            item["speaker_source"] = item.get("speaker_source", "srt_hint")

        item["speech_seconds"] = round(max(float(item["end"]) - float(item["start"]), 0.0), 3)
        updated.append(item)

    return updated


def identify_speakers_vllm(
    segments: list[dict],
    *,
    cache_path: Path | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fallback: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Label segments with speakers via vLLM.
    Returns (updated_segments, speaker_summary).
    """
    if not segments:
        return segments, {}

    config = vllm_config(base_url, model, api_key)
    cache_path = cache_path or Path("speaker_vllm.cache.json")

    cached = _load_cache(cache_path)
    if cached is not None and all(index in cached for index in range(len(segments))):
        print(f"Loaded vLLM speaker labels from cache: {cache_path}")
        labeled = _apply_assignments(segments, cached)
        summary = speaker_speech_summary(labeled)
        print_speaker_summary(summary)
        return labeled, summary

    cast_context = build_cast_context()
    assignments: dict[int, dict] = dict(cached or {})
    pending = [index for index in range(len(segments)) if index not in assignments]

    if pending:
        print(
            f"Identifying speakers with vLLM ({config['base_url']}, "
            f"batch={batch_size}, pending={len(pending)})..."
        )

    try:
        for offset in range(0, len(pending), batch_size):
            batch_indices = pending[offset : offset + batch_size]
            batch_payload = [_segment_payload(index, segments[index]) for index in batch_indices]
            prompt = _build_prompt(batch_payload, cast_context)
            content = vllm_chat(
                prompt,
                base_url=config["base_url"],
                model=config["model"],
                api_key=config["api_key"],
            )
            parsed = _extract_json_object(content)
            for item in parsed.get("segments", []):
                assignments[int(item["index"])] = item

            partial = [_apply_assignments(segments, assignments)[i] for i in batch_indices]
            _save_cache(
                cache_path,
                [assignments[i] for i in sorted(assignments)],
                speaker_speech_summary(partial),
            )
            print(f"  vLLM labeled {min(offset + batch_size, len(pending))}/{len(pending)} segments")
    except Exception as exc:
        if not fallback:
            raise
        print(f"Warning: vLLM speaker ID failed ({exc}); falling back to heuristic speaker propagation.")
        segments = propagate_speaker(segments)
        summary = speaker_speech_summary(segments)
        print_speaker_summary(summary)
        return segments, summary

    labeled = _apply_assignments(segments, assignments)
    labeled = propagate_speaker(labeled)
    summary = speaker_speech_summary(labeled)
    _save_cache(
        cache_path,
        [assignments[i] for i in sorted(assignments)],
        summary,
    )
    print_speaker_summary(summary)
    return labeled, summary
