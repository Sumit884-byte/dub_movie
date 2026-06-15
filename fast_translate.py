"""Fast translation: parallel multi-backend sharding + batch cloud + Google."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from deep_translator import GoogleTranslator

DEFAULT_CLOUD_URL = os.environ.get("DUB_TRANSLATE_CLOUD_URL", "http://localhost:11434/v1")
DEFAULT_CLOUD_MODEL = os.environ.get("DUB_TRANSLATE_CLOUD_MODEL", "minimax-m2.5:cloud")
DEFAULT_BATCH_SIZE = int(os.environ.get("DUB_TRANSLATE_BATCH", "10"))
DEFAULT_WORKERS = int(os.environ.get("DUB_TRANSLATE_WORKERS", "16"))
DEFAULT_TIMEOUT = float(os.environ.get("DUB_TRANSLATE_TIMEOUT", "120"))
DEFAULT_MAX_RETRIES = int(os.environ.get("DUB_TRANSLATE_MAX_RETRIES", "4"))
DEFAULT_RETRY_BASE_SEC = float(os.environ.get("DUB_TRANSLATE_RETRY_BASE_SEC", "2.0"))
DEFAULT_PARALLEL_BACKENDS = os.environ.get(
    "DUB_TRANSLATE_PARALLEL_BACKENDS", "minimax,google"
)

_TARGET_NAMES = {
    "hi": "Hindi",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ko": "Korean",
}

_tls = threading.local()


def _lang_key(language: str) -> str:
    return language.split("-", 1)[0].lower()


def _google_client(target_language: str) -> GoogleTranslator:
    lang = _lang_key(target_language)
    clients: dict[str, GoogleTranslator] = getattr(_tls, "google_clients", {})
    if lang not in clients:
        clients[lang] = GoogleTranslator(source="ja", target=lang)
        _tls.google_clients = clients
    return clients[lang]


def _retry_delay(attempt: int) -> float:
    return DEFAULT_RETRY_BASE_SEC * (2 ** attempt)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in (429, 503, 529):
            return True
        text = (exc.response.text or text).lower()
    markers = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota",
        "429",
        "503",
        "subscription",
        "capacity",
        "overloaded",
    )
    return any(marker in text for marker in markers)


def google_translate_with_retry(text: str, target_language: str) -> str:
    text = text.strip()
    if not text:
        return ""
    last_exc: Exception | None = None
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            return _google_client(target_language).translate(text)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= DEFAULT_MAX_RETRIES:
                break
            delay = _retry_delay(attempt)
            if _is_rate_limit_error(exc):
                delay *= 2
            print(f"Google retry {attempt + 2}/{DEFAULT_MAX_RETRIES} in {delay:.1f}s ({exc})")
            time.sleep(delay)
    raise last_exc or RuntimeError("Google translation failed")


def google_translate(text: str, target_language: str) -> str:
    return google_translate_with_retry(text, target_language)


def google_translate_parallel(
    texts: list[str],
    target_language: str,
    *,
    workers: int | None = None,
) -> list[str]:
    if not texts:
        return []
    workers = min(workers or DEFAULT_WORKERS, len(texts))

    def _one(text: str) -> str:
        try:
            return google_translate_with_retry(text, target_language)
        except Exception as exc:
            print(f"Google line failed: {exc}")
            return ""

    results: list[str | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, text): index for index, text in enumerate(texts)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return [item or "" for item in results]


def _resolve_parallel_backends(
    target_language: str,
    *,
    backends: str | None = None,
) -> list[str]:
    """Pick backends that are actually available for parallel sharding."""
    del target_language
    requested = [
        name.strip().lower()
        for name in (backends or DEFAULT_PARALLEL_BACKENDS).split(",")
        if name.strip()
    ]
    available: list[str] = []
    for name in requested:
        if name in ("minimax", "cloud") and "minimax" not in available:
            available.append("minimax")
        elif name == "google" and "google" not in available:
            available.append("google")
    return available or ["google"]


def _translate_shard(
    texts: list[str],
    backend: str,
    target_language: str,
    *,
    cloud_url: str | None = None,
    cloud_model: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
) -> list[str]:
    if not texts:
        return []
    if backend == "minimax":
        return translate_batch_cloud(
            texts,
            target_language,
            batch_size=batch_size,
            base_url=cloud_url,
            model=cloud_model,
        )
    if backend == "google":
        return google_translate_parallel(texts, target_language, workers=workers)
    raise ValueError(f"Unknown shard backend: {backend}")


def _refill_missing(
    texts: list[str],
    indices: list[int],
    results: list[str | None],
    target_language: str,
    *,
    backend: str,
    cloud_url: str | None = None,
    cloud_model: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Fill empty slots using a backup backend."""
    pending = [index for index in indices if not results[index] and texts[index].strip()]
    if not pending:
        return
    print(f"Refilling {len(pending)} line(s) via {backend} backup...")
    try:
        if backend == "google":
            filled = google_translate_parallel(
                [texts[i] for i in pending],
                target_language,
                workers=workers,
            )
        elif backend == "minimax":
            filled = translate_batch_cloud(
                [texts[i] for i in pending],
                target_language,
                batch_size=batch_size,
                base_url=cloud_url,
                model=cloud_model,
            )
        else:
            return
        for index, translated in zip(pending, filled):
            if translated and str(translated).strip():
                results[index] = str(translated).strip()
    except Exception as exc:
        print(f"{backend} refill failed ({exc}).")


def translate_many_parallel(
    texts: list[str],
    target_language: str,
    *,
    cloud_url: str | None = None,
    cloud_model: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
    parallel_backends: str | None = None,
) -> list[str]:
    """
    Split lines across Minimax and Google — all shards run at once.
    Missing lines are cross-refilled from the other cloud backend.
    """
    if not texts:
        return []

    backends = _resolve_parallel_backends(target_language, backends=parallel_backends)
    shard_count = len(backends)
    chunk_size = max(1, math.ceil(len(texts) / shard_count))

    plan: list[str] = []
    for shard_idx, backend in enumerate(backends):
        start = shard_idx * chunk_size
        end = min(len(texts), start + chunk_size)
        if start < end:
            plan.append(f"{backend} ({end - start})")
    print(f"Parallel translation: {len(texts)} lines -> {', '.join(plan)}")

    results: list[str | None] = [None] * len(texts)
    jobs: list[tuple[str, list[str], list[int]]] = []
    for shard_idx, backend in enumerate(backends):
        start = shard_idx * chunk_size
        end = min(len(texts), start + chunk_size)
        if start >= end:
            continue
        indices = list(range(start, end))
        jobs.append((backend, texts[start:end], indices))

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(
                _translate_shard,
                shard_texts,
                backend,
                target_language,
                cloud_url=cloud_url,
                cloud_model=cloud_model,
                workers=workers,
                batch_size=batch_size,
            ): (backend, indices)
            for backend, shard_texts, indices in jobs
        }
        for future in as_completed(futures):
            backend, indices = futures[future]
            try:
                shard_out = future.result()
            except Exception as exc:
                print(f"Shard {backend} failed ({exc}); will refill from backup.")
                continue
            for index, translated in zip(indices, shard_out):
                if translated and translated.strip():
                    results[index] = translated.strip()

    refill_backends: list[str] = ["google", "minimax"]
    all_indices = list(range(len(texts)))
    for backend in refill_backends:
        missing = [i for i in all_indices if not results[i] and texts[i].strip()]
        if not missing:
            break
        _refill_missing(
            texts,
            missing,
            results,
            target_language,
            backend=backend,
            cloud_url=cloud_url,
            cloud_model=cloud_model,
            workers=workers,
            batch_size=batch_size,
        )

    unresolved = sum(1 for i, text in enumerate(texts) if not results[i] and text.strip())
    if unresolved:
        print(f"Warning: {unresolved} line(s) could not be translated after all backups.")

    return [item or "" for item in results]


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


def minimax_batch_translate(
    texts: list[str],
    target_language: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    """Translate a batch of lines with Minimax via Ollama cloud."""
    if not texts:
        return []

    lang = _lang_key(target_language)
    lang_name = _TARGET_NAMES.get(lang, lang)
    base_url = (base_url or DEFAULT_CLOUD_URL).rstrip("/")
    model = model or DEFAULT_CLOUD_MODEL

    payload_items = [{"id": index, "ja": text} for index, text in enumerate(texts)]
    prompt = (
        f"Translate each Japanese anime subtitle line to {lang_name}.\n"
        "Keep names like Nobita/Doraemon as spoken Hindi forms when natural.\n"
        'Return JSON only: {"lines":[{"id":0,"hi":"..."}]}\n'
        f"Input:\n{json.dumps(payload_items, ensure_ascii=False)}"
    )
    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You translate anime subtitles accurately. "
                    "Reply with valid JSON only, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max(512, 180 * len(texts)),
    }

    last_exc: Exception | None = None
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                json=request,
                timeout=timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = _extract_json_object(content)
            lines = parsed.get("lines", parsed.get("translations", []))

            out = [""] * len(texts)
            text_key = lang if lang in ("hi", "en") else "text"
            for item in lines:
                index = int(item.get("id", item.get("index", -1)))
                if 0 <= index < len(texts):
                    out[index] = str(item.get(text_key) or item.get("text") or "").strip()
            return out
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= DEFAULT_MAX_RETRIES:
                break
            delay = _retry_delay(attempt)
            if _is_rate_limit_error(exc):
                delay *= 2
                print(
                    f"Minimax rate limit/backoff — retry {attempt + 2}/{DEFAULT_MAX_RETRIES} "
                    f"in {delay:.1f}s"
                )
            else:
                print(f"Minimax retry {attempt + 2}/{DEFAULT_MAX_RETRIES} in {delay:.1f}s ({exc})")
            time.sleep(delay)

    raise last_exc or RuntimeError("Minimax batch translation failed")


def translate_batch_cloud(
    texts: list[str],
    target_language: str,
    *,
    batch_size: int | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> list[str]:
    """Minimax batched cloud translation with Google fallback per missing line."""
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    results = [""] * len(texts)

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        try:
            translated = minimax_batch_translate(
                chunk,
                target_language,
                base_url=base_url,
                model=model,
            )
        except Exception as exc:
            print(f"Minimax batch failed ({exc}); falling back to Google for chunk.")
            translated = google_translate_parallel(chunk, target_language)

        for offset, text in enumerate(translated):
            index = start + offset
            if text:
                results[index] = text
            elif chunk[offset].strip():
                try:
                    results[index] = google_translate_with_retry(chunk[offset], target_language)
                except Exception as exc:
                    print(f"Google fallback failed for line {index}: {exc}")
                    try:
                        results[index] = minimax_batch_translate(
                            [chunk[offset]],
                            target_language,
                            base_url=base_url,
                            model=model,
                        )[0]
                    except Exception as mini_exc:
                        print(f"Minimax single-line fallback failed for {index}: {mini_exc}")

    return results


def translate_text(
    text: str,
    target_language: str,
    *,
    backend: str = "auto",
    cloud_url: str | None = None,
    cloud_model: str | None = None,
) -> str:
    """Translate one line using Minimax cloud and/or Google."""
    text = text.strip()
    if not text:
        return ""

    backend = backend.lower()

    if backend in ("minimax", "cloud"):
        try:
            return minimax_batch_translate([text], target_language, base_url=cloud_url, model=cloud_model)[0]
        except Exception as exc:
            print(f"Minimax failed ({exc}); using Google.")
            return google_translate(text, target_language)

    if backend == "google":
        return google_translate(text, target_language)

    # auto: cloud first, then Google
    try:
        return minimax_batch_translate([text], target_language, base_url=cloud_url, model=cloud_model)[0]
    except Exception:
        pass

    return google_translate(text, target_language)


def translate_many(
    texts: list[str],
    target_language: str,
    *,
    backend: str = "auto",
    cloud_url: str | None = None,
    cloud_model: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
    parallel_backends: str | None = None,
    **_: object,
) -> list[str]:
    """Fast translation for many subtitle lines (Minimax cloud + Google only)."""
    if not texts:
        return []

    backend = backend.lower()

    if backend == "parallel":
        return translate_many_parallel(
            texts,
            target_language,
            cloud_url=cloud_url,
            cloud_model=cloud_model,
            workers=workers,
            batch_size=batch_size,
            parallel_backends=parallel_backends,
        )

    if backend == "auto":
        parallel = _resolve_parallel_backends(target_language, backends=parallel_backends)
        if len(parallel) >= 2:
            return translate_many_parallel(
                texts,
                target_language,
                cloud_url=cloud_url,
                cloud_model=cloud_model,
                workers=workers,
                batch_size=batch_size,
                parallel_backends=parallel_backends,
            )

    if backend in ("minimax", "cloud", "auto"):
        try:
            return translate_batch_cloud(
                texts,
                target_language,
                batch_size=batch_size,
                base_url=cloud_url,
                model=cloud_model,
            )
        except Exception as exc:
            if backend in ("minimax", "cloud"):
                print(f"Cloud batch translation failed ({exc}); falling back to Google.")
            elif backend == "auto":
                print(f"Auto: Minimax unavailable ({exc}); using Google parallel.")

    if backend in ("google", "auto", "minimax", "cloud"):
        return google_translate_parallel(texts, target_language, workers=workers)

    raise RuntimeError(f"Unknown translation backend: {backend}")
