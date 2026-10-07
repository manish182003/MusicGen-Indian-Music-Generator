# MusicGen Indian Music Generation API

Fine-tuned MusicGen model with LoRA adapter for generating Indian classical and instrumental music from text prompts.

## Project Overview

- **Base Model**: `facebook/musicgen-small`
- **Adapter Location**: `models/adapters/adapter3`
- **Fine-Tuning Notebook**: `notebooks/audio_fine_tuning.ipynb`
- **Output Directory**: `outputs/`
- **FastAPI Application**: `src/api/main.py`

## How to Start FastAPI Server

Start the API server using Uvicorn:

```bash
uvicorn src.api.main:app --reload
```

The server runs at `http://127.0.0.1:8000`.

## API Endpoints

### 1. Health Check
- **GET** `http://127.0.0.1:8000/health`
- Response:
  ```json
  {
      "status": "ok"
  }
  ```

### 2. Generate Music
- **POST** `http://127.0.0.1:8000/generate`
- Request Body:
  ```json
  {
      "prompt": "Indian classical sitar with tabla and tanpura drone",
      "max_new_tokens": 1500,
      "temperature": 1.0,
      "guidance_scale": 3.5,
      "top_k": 250
  }
  ```
- Response:
  ```json
  {
      "status": "completed",
      "file": "outputs/generated_1750000000_a1b2c3.wav",
      "filename": "generated_1750000000_a1b2c3.wav",
      "sampling_rate": 32000,
      "duration_seconds": 30.0
  }
  ```

### 3. Retrieve Audio File
- **GET** `http://127.0.0.1:8000/audio/{filename}`
- Example: `http://127.0.0.1:8000/audio/generated_1750000000_a1b2c3.wav`
- Returns the WAV audio file stream.
