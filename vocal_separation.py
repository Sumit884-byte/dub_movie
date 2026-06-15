"""Vocal / instrumental separation using Demucs (Meta)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def separate_with_demucs(
    input_wav: Path,
    output_wav: Path,
    work_dir: Path | None = None,
    model: str = "htdemucs",
    force: bool = False,
) -> Path:
    """
    Run Demucs two-stems mode and return the instrumental (no_vocals) path.
    Writes the final instrumental track to output_wav.
    """
    if not force and output_wav.exists() and output_wav.stat().st_size > 0:
        print(f"Using existing Demucs instrumental: {output_wav}")
        return output_wav

    work_dir = work_dir or output_wav.parent / "demucs_out"
    work_dir.mkdir(parents=True, exist_ok=True)
    stem_name = input_wav.stem
    instrumental = work_dir / model / stem_name / "no_vocals.wav"

    if not force and instrumental.exists() and instrumental.stat().st_size > 0:
        _copy_wav(instrumental, output_wav)
        return output_wav

    print(f"Running Demucs ({model}) vocal separation on {input_wav.name}...")
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "-o", str(work_dir),
        str(input_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs failed:\n{result.stderr[-2000:] if result.stderr else result.stdout}"
        )

    if not instrumental.exists():
        candidates = list(work_dir.rglob("no_vocals.wav"))
        if not candidates:
            raise RuntimeError(f"Demucs finished but no_vocals.wav not found under {work_dir}")
        instrumental = candidates[0]

    _copy_wav(instrumental, output_wav)
    print(f"Demucs instrumental saved: {output_wav}")
    return output_wav


def _copy_wav(source: Path, dest: Path) -> None:
    if source.resolve() == dest.resolve():
        return
    dest.write_bytes(source.read_bytes())
