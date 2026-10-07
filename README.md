---
title: Indian Music Generator (MusicGen + LoRA)
emoji: 🎵
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.16.0"
app_file: app.py
pinned: false
---

# 🎵 Indian Music Generator (MusicGen + PEFT LoRA)

[![Hugging Face Spaces](https://img.shields.io/badge/-%F0%9F%A4%97%20Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/manish182/MusicGen-FineTuned)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-App-brightgreen)](https://manish182-musicgen-finetuned.hf.space)
[![Model](https://img.shields.io/badge/Base%20Model-facebook%2Fmusicgen--small-orange)](https://huggingface.co/facebook/musicgen-small)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-PEFT%20LoRA-purple)](https://github.com/huggingface/peft)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Gradio%205-green)](https://fastapi.tiangolo.com/)

A production-grade Deep Learning system for generating high-fidelity **Indian Classical & Instrumental Music** from text prompts. Built by fine-tuning Meta AI's **`facebook/musicgen-small`** transformer model using **Parameter-Efficient Fine-Tuning (PEFT) LoRA**, served via **FastAPI** backend and an interactive **Gradio 5** UI on **Hugging Face Spaces**.

---

## 🌟 Live Demos

- 🚀 **Hugging Face Space**: [manish182/MusicGen-FineTuned](https://huggingface.co/spaces/manish182/MusicGen-FineTuned)
- 🌐 **Full-Screen Web Application**: [manish182-musicgen-finetuned.hf.space](https://manish182-musicgen-finetuned.hf.space)

---

## 📐 System Architecture

```text
[ Text Prompt / Settings ]
          │
          ▼
   ┌──────────────┐
   │  Gradio UI   │  <--->  [ FastAPI REST API (/generate) ]
   └──────┬───────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              MusicGenerator Inference Engine            │
│                                                         │
│  ┌────────────────────────┐   ┌──────────────────────┐  │
│  │ facebook/musicgen-small│ + │ PEFT LoRA Adapter    │  │
│  │ (Base Audio Transformer)│   │ (models/adapters/    │  │
│  │                        │   │  adapter3)           │  │
│  └────────────────────────┘   └──────────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
            [ 32kHz WAV Audio Output ]
```

---

## 📊 Data Gathering & Audio Preprocessing Pipeline

### 1. Dataset Collection
The training corpus consists of curated high-quality Indian instrumental compositions featuring:
- **Melodic Instruments**: Sitar, Bansuri (Bamboo Flute), Santoor, Carnatic Violin, Veena, Sarangi.
- **Rhythmic Percussion**: Tabla, Mridangam, Pakhawaj, Ghatam.
- **Drone Instruments**: Tanpura (Tambura).

### 2. Audio Processing Pipeline (`src/preprocessing/`)
- **Resampling**: All audio tracks were standardized to **32,000 Hz (32 kHz)** sampling rate using `librosa` and `resampy`.
- **Channel Normalization**: Stereo tracks converted to **mono** to align with MusicGen's EnCodec audio compression format.
- **Chunking**: Continuous audio split into non-overlapping **30-second clips**.
- **Amplitude Normalization**: Peak normalization applied to ensure consistent volume levels across clips.

### 3. Text Prompt Structuring
Each audio chunk was paired with detailed descriptive text metadata following a structured prompt template:
> `"[Instrument] solo/duet with [Percussion] and [Drone], [Mood/Raga] instrumental melody"`

---

## 🧠 Model Fine-Tuning Methodology

Fine-tuning details documented in [`notebooks/audio_fine_tuning.ipynb`](notebooks/audio_fine_tuning.ipynb):

- **Base Architecture**: `facebook/musicgen-small` (300 Million parameters)
  - Text Encoder: T5 Transformer
  - Audio Decoder: 32kHz EnCodec model
- **LoRA Configuration**:
  - **Rank ($r$)**: `16`
  - **Scaling Factor ($\alpha$)**: `32`
  - **Target Modules**: `q_proj`, `v_proj` inside `self_attn` and `encoder_attn` projection layers.
  - **Dropout**: `0.0`
  - **Trainable Parameters**: ~4.2M parameters (< 1.5% of total base model weights).
- **Optimization**:
  - Optimizer: `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay `0.01`)
  - Learning Rate: `1e-4` with cosine lr scheduler.
  - Precision: Automatic Mixed Precision (`fp16`).

---

## 📂 Project Structure

```text
MusicGen/
├── app.py                       # Gradio 5 Web Interface for HF Spaces
├── requirements.txt             # Production dependency manifest
├── README.md                    # Project documentation & HF Space config
├── .gitignore                   # Excluded raw datasets, audio outputs & envs
├── .gitattributes              # Git LFS tracking configuration (*.safetensors)
├── config/
│   └── default_config.yaml      # Model hyperparameter configurations
├── models/
│   └── adapters/
│       └── adapter3/            # Fine-tuned PEFT LoRA weights
│           ├── adapter_config.json
│           └── adapter_model.safetensors
├── notebooks/
│   └── audio_fine_tuning.ipynb  # End-to-end model fine-tuning notebook
├── outputs/                     # Generated WAV audio destination
├── src/
│   ├── api/                     # FastAPI Application server
│   │   └── main.py              # REST API endpoints (/health, /generate, /audio)
│   └── inference/               # Model loading & inference engine
│       └── music_generator.py   # MusicGenerator wrapper class
└── tests/                       # Unit testing suite
    └── test_api.py              # PyTest test cases
```

---

## 🛠️ Local Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git LFS installed (`git lfs install`)

### 2. Clone & Setup Environment
```bash
git clone https://github.com/manish182003/MusicGen-Indian-Music-Generator.git
cd MusicGen-Indian-Music-Generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run FastAPI Backend Server
```bash
uvicorn src.api.main:app --reload
```
The REST API will be available at `http://127.0.0.1:8000`. Access interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 4. Run Gradio Web Application
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

### 5. Run Test Suite
```bash
pytest tests/
```

---

## 🔌 API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Generate Music
- **Endpoint**: `POST /generate`
- **Request Body**:
  ```json
  {
    "prompt": "Indian classical sitar with tabla and tanpura drone",
    "max_new_tokens": 1500,
    "temperature": 1.0,
    "guidance_scale": 3.5,
    "top_k": 250
  }
  ```
- **Response**:
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
- **Endpoint**: `GET /audio/{filename}`
- **Example**: `http://127.0.0.1:8000/audio/generated_1750000000_a1b2c3.wav`
- **Response**: WAV audio file stream (`audio/wav`).

---

## ⚡ Deployment & Execution Features

- **Hugging Face ZeroGPU Acceleration**: Automatically utilizes NVIDIA GPUs when available on Hugging Face Spaces.
- **Fail-Safe CPU Fallback**: Automatically catches GPU worker queue timeouts and seamlessly offloads inference to CPU without throwing client-facing errors.
- **Git LFS Enabled**: PEFT LoRA adapter weights (`adapter_model.safetensors`) are tracked using Git LFS for seamless repository synchronization.

---

## 📜 License & Acknowledgments

- **Base Model**: [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) by Meta AI (MIT License).
- **Libraries**: Hugging Face `transformers`, `peft`, `gradio`, `fastapi`.
