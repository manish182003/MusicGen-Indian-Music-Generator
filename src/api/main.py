import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.inference.music_generator import MusicGenerator

BASE_MODEL_NAME = os.getenv("MODEL_NAME", "facebook/musicgen-small")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/adapters/adapter3")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

generator: Optional[MusicGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that loads MusicGenerator model once on startup."""
    global generator
    print("Initializing MusicGenerator on startup...")
    generator = MusicGenerator(
        base_model_name=BASE_MODEL_NAME,
        adapter_path=ADAPTER_PATH,
    )
    if not Path(ADAPTER_PATH).exists():
        print(f"Warning: Specified ADAPTER_PATH '{ADAPTER_PATH}' does not exist.")
    try:
        generator.load_model()
    except Exception as exc:
        print(f"Startup warning loading model: {exc}")
    yield


app = FastAPI(
    title="MusicGen Indian Music Generation API",
    description="FastAPI endpoint for generating music using MusicGen and trained LoRA adapter.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Text prompt describing the desired music.",
        json_schema_extra={"example": "Indian classical sitar with tabla and tanpura drone"},
    )
    max_new_tokens: Optional[int] = Field(
        1500,
        ge=10,
        le=3000,
        description="Maximum new audio tokens to generate (1500 tokens ~ 30s).",
    )
    temperature: Optional[float] = Field(
        1.0,
        gt=0,
        le=2.0,
        description="Sampling temperature.",
    )
    guidance_scale: Optional[float] = Field(
        3.5,
        gt=0,
        le=10.0,
        description="Classifier-free guidance scale.",
    )
    top_k: Optional[int] = Field(
        250,
        gt=0,
        description="Top-k sampling parameter.",
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/generate")
def generate_music(request: GenerationRequest):
    """Generate audio WAV file from text prompt."""
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if generator is None:
        raise HTTPException(
            status_code=500,
            detail="Music generator is not initialized.",
        )

    try:
        result = generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens or 1500,
            temperature=request.temperature or 1.0,
            guidance_scale=request.guidance_scale or 3.5,
            top_k=request.top_k or 250,
            output_dir=OUTPUT_DIR,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Music generation failed: {str(exc)}",
        )


@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """Retrieve generated WAV audio file."""
    safe_filename = Path(filename).name
    file_path = Path(OUTPUT_DIR) / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=safe_filename,
    )
