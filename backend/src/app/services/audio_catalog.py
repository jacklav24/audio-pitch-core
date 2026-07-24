from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from app.schemas import AudioCatalog, AudioFileSummary


class AudioCatalogError(Exception):
    """Base error for audio catalog operations."""


class InvalidAudioSelection(AudioCatalogError):
    """Raised when a selection is not a valid WAV file."""


class AudioFileNotFound(AudioCatalogError):
    """Raised when a selected WAV file does not exist."""


def _wav_info(path: Path) -> Any | None:
    if path.suffix.lower() != ".wav":
        return None

    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        return None

    return info if info.format == "WAV" else None


def list_audio_files(directory: Path) -> AudioCatalog:
    if not directory.is_dir():
        raise AudioCatalogError(f"Audio directory does not exist: {directory}")

    files: list[AudioFileSummary] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue

        info = _wav_info(path)
        if info is None:
            continue

        files.append(
            AudioFileSummary(
                name=path.name,
                size_bytes=path.stat().st_size,
                duration_seconds=float(info.duration),
            )
        )

    return AudioCatalog(directory=str(directory), files=files)


def resolve_audio_file(directory: Path, filename: str) -> Path:
    requested_name = Path(filename)
    if (
        not filename
        or requested_name.name != filename
        or requested_name.suffix.lower() != ".wav"
    ):
        raise InvalidAudioSelection("Only .wav files from the audio directory are allowed.")

    candidate = (directory / filename).resolve()
    if candidate.parent != directory.resolve():
        raise InvalidAudioSelection("The selected file is outside the audio directory.")

    if not candidate.is_file():
        raise AudioFileNotFound(f"Audio file not found: {filename}")

    if _wav_info(candidate) is None:
        raise InvalidAudioSelection("The selected file is not a valid WAV file.")

    return candidate

