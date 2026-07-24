from pydantic import BaseModel


class AudioFileSummary(BaseModel):
    name: str
    size_bytes: int
    duration_seconds: float


class AudioCatalog(BaseModel):
    directory: str
    files: list[AudioFileSummary]


class PitchPoint(BaseModel):
    time_seconds: float
    f0_hz: float | None
    confidence: float


class PitchAnalysis(BaseModel):
    filename: str
    duration_seconds: float
    sample_rate: int
    frame_size_ms: float
    hop_size_ms: float
    f_min_hz: float
    f_max_hz: float
    points: list[PitchPoint]

