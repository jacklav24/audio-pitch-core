from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import analysis.pitch_frame as pitch
import core.audio_buffer as audio_buffers
import core.framing as framing

from app.schemas import PitchAnalysis, PitchPoint


FRAME_SIZE_SECONDS = 0.1
HOP_SIZE_SECONDS = 0.025
F_MIN_HZ = 30.0
F_MAX_HZ = 500.0


class PitchAnalysisError(Exception):
    """Raised when a WAV file cannot be analyzed."""


def analyze_pitch(path: Path) -> PitchAnalysis:
    stat = path.stat()
    return _analyze_pitch_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _analyze_pitch_cached(
    path_value: str,
    modified_time_ns: int,
    file_size: int,
) -> PitchAnalysis:
    del modified_time_ns, file_size
    path = Path(path_value)
    audio = audio_buffers.load_audio_buffer(str(path))
    if audio is None:
        raise PitchAnalysisError(f"Could not load WAV file: {path.name}")

    sample_rate = audio.get_sample_rate()
    frame_size_samples = int(FRAME_SIZE_SECONDS * sample_rate)
    hop_size_samples = int(HOP_SIZE_SECONDS * sample_rate)
    frames = list(
        framing.build_frames(audio, frame_size_samples, hop_size_samples)
    )
    pitch_frames = list(
        pitch.estimate_pitch_sequence(
            frames,
            sample_rate,
            method="autocorr",
            f_min=F_MIN_HZ,
            f_max=F_MAX_HZ,
        )
    )

    points = [
        PitchPoint(
            time_seconds=frame.time_seconds,
            f0_hz=pitch_frame.f0_hz,
            confidence=pitch_frame.confidence,
        )
        for frame, pitch_frame in zip(frames, pitch_frames)
    ]

    return PitchAnalysis(
        filename=path.name,
        duration_seconds=len(audio.get_data()) / sample_rate,
        sample_rate=sample_rate,
        frame_size_ms=FRAME_SIZE_SECONDS * 1000,
        hop_size_ms=HOP_SIZE_SECONDS * 1000,
        f_min_hz=F_MIN_HZ,
        f_max_hz=F_MAX_HZ,
        points=points,
    )

