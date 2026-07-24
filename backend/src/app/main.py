from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import AudioCatalog, PitchAnalysis
from app.services.audio_catalog import (
    AudioCatalogError,
    AudioFileNotFound,
    InvalidAudioSelection,
    list_audio_files,
    resolve_audio_file,
)
from app.services.pitch_analysis import PitchAnalysisError, analyze_pitch


app = FastAPI(
    title="Audio Pitch Core",
    description="Local API for inspectable audio pitch analysis.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/audio-files", response_model=AudioCatalog)
def audio_files() -> AudioCatalog:
    try:
        return list_audio_files(get_settings().audio_directory)
    except AudioCatalogError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/pitch", response_model=PitchAnalysis)
async def pitch_analysis(filename: str) -> PitchAnalysis:
    settings = get_settings()
    try:
        path = resolve_audio_file(settings.audio_directory, filename)
        return await run_in_threadpool(analyze_pitch, path)
    except InvalidAudioSelection as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except AudioFileNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PitchAnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

