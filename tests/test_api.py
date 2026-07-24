from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def _write_sine_wave(path: Path, frequency: float = 110.0) -> None:
    sample_rate = 8_000
    duration_seconds = 0.5
    times = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    samples = (0.5 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def test_catalog_lists_only_valid_wav_files(tmp_path: Path, monkeypatch) -> None:
    _write_sine_wave(tmp_path / "bass.wav")
    (tmp_path / "bass.mp3").write_bytes(b"not an audio file")
    (tmp_path / "renamed.wav").write_bytes(b"not a wav file")
    monkeypatch.setenv("AUDIO_FILES_DIR", str(tmp_path))

    response = TestClient(app).get("/api/audio-files")

    assert response.status_code == 200
    assert [item["name"] for item in response.json()["files"]] == ["bass.wav"]


def test_pitch_endpoint_uses_existing_autocorrelation_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_sine_wave(tmp_path / "bass.wav")
    monkeypatch.setenv("AUDIO_FILES_DIR", str(tmp_path))

    response = TestClient(app).get("/api/pitch", params={"filename": "bass.wav"})

    assert response.status_code == 200
    payload = response.json()
    voiced = [point["f0_hz"] for point in payload["points"] if point["f0_hz"]]
    assert payload["filename"] == "bass.wav"
    assert len(payload["points"]) == 17
    assert voiced
    assert np.median(voiced) == pytest.approx(110.0, abs=2.0)


def test_pitch_endpoint_rejects_mp3_and_path_traversal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUDIO_FILES_DIR", str(tmp_path))
    client = TestClient(app)

    assert client.get("/api/pitch", params={"filename": "bass.mp3"}).status_code == 400
    assert client.get("/api/pitch", params={"filename": "../bass.wav"}).status_code == 400

