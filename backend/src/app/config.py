from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    audio_directory: Path
    frontend_directory: Path


def get_settings() -> Settings:
    configured_audio_directory = Path(
        os.environ.get("AUDIO_FILES_DIR", "bass_files")
    ).expanduser()

    if not configured_audio_directory.is_absolute():
        configured_audio_directory = PROJECT_ROOT / configured_audio_directory

    return Settings(
        project_root=PROJECT_ROOT,
        audio_directory=configured_audio_directory.resolve(),
        frontend_directory=PROJECT_ROOT / "frontend",
    )

