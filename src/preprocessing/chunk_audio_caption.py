import os
import sys
import json
import re
import time
import random
import base64
import tempfile
import traceback
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from google import genai
from groq import Groq
from openai import OpenAI


# =============================================================
# ENCODING FIX FOR WINDOWS CONSOLE
# =============================================================

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =============================================================
# ENVIRONMENT
# =============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# =============================================================
# CLIENTS
# =============================================================

openai_client = None
groq_client = None
gemini_client = None
openrouter_client = None


if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=25.0,
            max_retries=0,
        )
    except Exception as exc:
        print(f"[INIT] OpenAI client init warning: {exc}")


if GROQ_API_KEY:
    try:
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=25.0,
            max_retries=0,
        )
    except Exception as exc:
        print(f"[INIT] Groq client init warning: {exc}")


if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception as exc:
        print(f"[INIT] Gemini client init warning: {exc}")


# OpenRouter uses an OpenAI-compatible API.
if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=25.0,
            max_retries=0,
        )
    except Exception as exc:
        print(f"[INIT] OpenRouter client init warning: {exc}")


# =============================================================
# CONFIGURATION & MODEL POOLS
# =============================================================

# -------------------------------------------------------------
# GEMINI MODEL FALLBACK CHAIN
# -------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

# -------------------------------------------------------------
# GROQ MULTI-MODEL FALLBACK
# -------------------------------------------------------------
GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Qwen 3.8 supports mel-spectrogram image input.
GROQ_VISION_MODELS = {
    "qwen/qwen3.8-27b",
}

GROQ_MAX_COMPLETION_TOKENS = 700


# -------------------------------------------------------------
# OPENROUTER MULTI-MODEL FALLBACK
# -------------------------------------------------------------
OPENROUTER_MODELS = [
    "qwen/qwen3.8-27b:free",
    "google/gemini-2.0-flash-lite:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
]

OPENROUTER_MAX_OUTPUT_TOKENS = 350


# -------------------------------------------------------------
# OPENAI
# -------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_OUTPUT_TOKENS = 500
OPENAI_ENABLED = False  # Disabled by default unless active billing confirmed


# -------------------------------------------------------------
# PACING & DELAYS
# -------------------------------------------------------------
DELAY_SECONDS = 1.0


# =============================================================
# MODEL COOLDOWN & CIRCUIT BREAKER SYSTEM
# =============================================================
# Instead of disabling models permanently on rate limits (429),
# models are put on temporary cooldown. They automatically become
# available again once the cooldown expires.

COOLDOWN_TIMESTAMPS = {}
PERMANENTLY_DISABLED = set()


def is_model_available(provider, model_name):
    key = (provider, model_name)
    if key in PERMANENTLY_DISABLED:
        return False
    expiry = COOLDOWN_TIMESTAMPS.get(key, 0)
    if time.time() < expiry:
        return False
    return True


def set_model_cooldown(provider, model_name, duration_seconds=60.0):
    key = (provider, model_name)
    COOLDOWN_TIMESTAMPS[key] = time.time() + duration_seconds
    print(f"[{provider.upper()}] Temporary cooldown for '{model_name}' ({duration_seconds:.0f}s)")


def set_model_disabled(provider, model_name, reason="Permanent failure"):
    key = (provider, model_name)
    PERMANENTLY_DISABLED.add(key)
    print(f"[{provider.upper()}] PERMANENTLY DISABLED '{model_name}': {reason}")


def parse_retry_after(error_text):
    """Extract retry delay seconds from error message if provided by API."""
    text = str(error_text)
    # Match patterns like "try again in 11.55s" or "try again in 12s"
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    # Match "retry-after: 12"
    match = re.search(r"retry[-_]after\s*[:=]?\s*([0-9]+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return None


def execute_with_retry(api_fn, provider, model_name, max_retries=2, base_delay=2.0):
    """
    Executes api_fn() with automatic retry on 429 rate limit errors or 5xx server issues.
    If retries fail, puts the model on temporary cooldown and raises exception to fallback.
    """
    if not is_model_available(provider, model_name):
        raise RuntimeError(f"Model {provider}:{model_name} is on cooldown or disabled.")

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return api_fn()
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()

            # Check for permanent account or model errors (401, 403, 404)
            if any(code in err_msg for code in ["401", "403", "404", "invalid_api_key", "billing_not_active", "not_found", "model_not_found"]):
                set_model_disabled(provider, model_name, reason=str(e))
                raise e

            # Check for rate limit (429) or transient server errors
            is_rate_limit = any(term in err_msg for term in ["429", "rate limit", "rate_limit", "quota", "too many requests", "resourceexhausted", "itpm", "tpm", "rpm"])
            is_transient = any(code in err_msg for code in ["500", "502", "503", "504", "timeout", "connection"])

            if is_rate_limit or is_transient:
                retry_after = parse_retry_after(err_msg)
                if attempt < max_retries:
                    sleep_time = retry_after if retry_after else (base_delay * (2 ** attempt) + random.uniform(0.5, 1.5))
                    print(f"[{provider.upper()}] Rate limit/transient issue on '{model_name}'. Retrying in {sleep_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                else:
                    cooldown = max(60.0, retry_after if retry_after else 60.0)
                    set_model_cooldown(provider, model_name, duration_seconds=cooldown)
                    raise e
            else:
                set_model_cooldown(provider, model_name, duration_seconds=30.0)
                raise e

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Execution failed for model '{provider}:{model_name}'")



# =============================================================
# OUTPUT FORMAT & PROMPTS
# =============================================================

EMPTY_RESULT = {
    "genre": "",
    "style": "",
    "mood": [],
    "instruments": [],
    "tempo_bpm": None,
    "rhythm": "",
    "structure": "",
    "full_description": "",
    "musicgen_prompt": "",
}


ANALYSIS_INSTRUCTION = """
You are an expert music metadata annotator.

You are analyzing ONE specific 30-second music clip.

Analyze ONLY this clip.

Do not use information from another part of the song.

Do not invent instruments.

Only identify instruments when the available evidence reasonably supports them.

Return ONLY valid JSON.

Use exactly this structure:

{
    "genre": "",
    "style": "",
    "mood": [],
    "instruments": [],
    "tempo_bpm": null,
    "rhythm": "",
    "structure": "",
    "full_description": "",
    "musicgen_prompt": ""
}

Rules:

- genre: broad musical genre
- style: musical tradition/style
- mood: audible moods
- instruments: reasonably supported instruments
- tempo_bpm: estimated BPM
- rhythm: rhythmic characteristics
- structure: what happens inside this 30-second clip
- full_description: concise description
- musicgen_prompt: concise prompt for instrumental MusicGen

The MusicGen prompt should describe the actual audible musical characteristics.

Do not mention acoustic measurements in the MusicGen prompt.

Return JSON only.
"""


# =============================================================
# JSON CLEANER & NORMALIZER
# =============================================================

def clean_json_response(text):
    if text is None:
        raise ValueError("Empty model response.")

    text = str(text).strip()

    if not text:
        raise ValueError("Empty model response.")

    # Remove common markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    # First try direct parsing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object from text (handles models that include reasoning text)
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        raise ValueError(
            f"Model response did not contain a JSON object: {text[:300]}"
        )

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON response: {error}. Response preview: {text[:300]}"
        ) from error


def normalize_result(result):
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object.")

    normalized = dict(EMPTY_RESULT)

    normalized["genre"] = str(result.get("genre", "")).strip()
    normalized["style"] = str(result.get("style", "")).strip()

    mood = result.get("mood", [])
    if isinstance(mood, list):
        normalized["mood"] = [str(m).strip() for m in mood if str(m).strip()]
    elif mood:
        normalized["mood"] = [str(mood).strip()]

    instruments = result.get("instruments", [])
    if isinstance(instruments, list):
        normalized["instruments"] = [str(i).strip() for i in instruments if str(i).strip()]
    elif instruments:
        normalized["instruments"] = [str(instruments).strip()]

    tempo = result.get("tempo_bpm", None)
    try:
        if tempo is not None:
            normalized["tempo_bpm"] = float(tempo)
    except Exception:
        normalized["tempo_bpm"] = None

    normalized["rhythm"] = str(result.get("rhythm", "")).strip()
    normalized["structure"] = str(result.get("structure", "")).strip()
    normalized["full_description"] = str(result.get("full_description", "")).strip()
    normalized["musicgen_prompt"] = str(result.get("musicgen_prompt", "")).strip()

    if not normalized["musicgen_prompt"]:
        # Synthesize fallback prompt if empty
        parts = []
        if normalized["genre"]:
            parts.append(normalized["genre"])
        if normalized["mood"]:
            parts.append(", ".join(normalized["mood"][:2]))
        if normalized["instruments"]:
            parts.append(", ".join(normalized["instruments"][:3]))
        if normalized["tempo_bpm"]:
            parts.append(f"{round(normalized['tempo_bpm'])} bpm")
        
        normalized["musicgen_prompt"] = ", ".join(parts) if parts else "instrumental music track with rhythmic drum pattern and atmospheric synths"

    return normalized


# =============================================================
# AUDIO PATH & FEATURE EXTRACTION
# =============================================================

def resolve_audio_path(audio_path, project_root):
    path_string = str(audio_path).replace("\\", "/")
    path = Path(path_string)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def analyze_audio_locally(audio_path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(librosa.feature.rms(y=y).mean())
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zero_crossing_rate = float(librosa.feature.zero_crossing_rate(y).mean())

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    try:
        tempo = float(tempo[0])
    except (TypeError, IndexError):
        tempo = float(tempo)

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = frequencies < 250
    mid_mask = (frequencies >= 250) & (frequencies < 2000)
    high_mask = frequencies >= 2000

    low_energy = float(stft[low_mask].mean())
    mid_energy = float(stft[mid_mask].mean())
    high_energy = float(stft[high_mask].mean())

    return {
        "sample_rate": int(sr),
        "duration_seconds": round(duration, 2),
        "rms": round(rms, 6),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "spectral_bandwidth_hz": round(spectral_bandwidth, 2),
        "spectral_rolloff_hz": round(spectral_rolloff, 2),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
        "estimated_tempo_bpm": round(tempo, 2),
        "low_frequency_energy": round(low_energy, 6),
        "mid_frequency_energy": round(mid_energy, 6),
        "high_frequency_energy": round(high_energy, 6),
    }


def create_spectrogram(audio_path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {audio_path.name}")
    plt.tight_layout()
    plt.savefig(temp_path, dpi=120)
    plt.close()

    return temp_path


# =============================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================

# -------------------------------------------------------------
# 1. GEMINI PROVIDER
# -------------------------------------------------------------

def caption_with_gemini_model(audio_path, model_name):
    if gemini_client is None:
        raise RuntimeError("Gemini client not initialized.")

    print(f"[GEMINI] Trying {model_name} on {audio_path.name}")
    uploaded_file = None
    try:
        uploaded_file = gemini_client.files.upload(file=str(audio_path))

        def call_api():
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, ANALYSIS_INSTRUCTION],
            )
            return response.text

        raw_text = execute_with_retry(call_api, "gemini", model_name)
        result = clean_json_response(raw_text)
        return normalize_result(result)
    finally:
        if uploaded_file is not None:
            try:
                gemini_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def caption_with_gemini(audio_path):
    if gemini_client is None:
        raise RuntimeError("Gemini client not initialized.")

    errors = []
    for model_name in GEMINI_MODELS:
        if not is_model_available("gemini", model_name):
            continue
        try:
            result = caption_with_gemini_model(audio_path, model_name)
            print(f"GEMINI SUCCESS: {model_name}")
            return result, f"gemini:{model_name}"
        except Exception as error:
            error_text = str(error)
            errors.append(f"{model_name}: {error_text}")
            print(f"GEMINI FAILED ({model_name}): {error_text[:120]}")
            continue

    raise RuntimeError("All Gemini models failed:\n" + "\n".join(errors))


# -------------------------------------------------------------
# 2. GROQ PROVIDER
# -------------------------------------------------------------

def caption_with_groq_model(audio_path, model_name):
    if groq_client is None:
        raise RuntimeError("Groq client not initialized.")

    print(f"[GROQ] Trying {model_name}")
    features = analyze_audio_locally(audio_path)
    prompt = f"{ANALYSIS_INSTRUCTION}\n\nObjective acoustic measurements:\n{json.dumps(features, indent=2)}"

    if model_name in GROQ_VISION_MODELS:
        spectrogram_path = create_spectrogram(audio_path)
        try:
            with open(spectrogram_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ]
        finally:
            spectrogram_path.unlink(missing_ok=True)
        extra_kwargs = {"reasoning_effort": "none"}
    else:
        messages = [{"role": "user", "content": prompt}]
        extra_kwargs = {"reasoning_effort": "low"}

    def call_api():
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            **extra_kwargs,
        )
        return completion.choices[0].message.content

    raw_text = execute_with_retry(call_api, "groq", model_name)
    result = clean_json_response(raw_text)
    result = normalize_result(result)

    if result["tempo_bpm"] is None:
        result["tempo_bpm"] = features["estimated_tempo_bpm"]

    return result


def caption_with_groq(audio_path):
    if groq_client is None:
        raise RuntimeError("Groq client not initialized.")

    errors = []
    for model_name in GROQ_MODELS:
        if not is_model_available("groq", model_name):
            continue
        try:
            result = caption_with_groq_model(audio_path, model_name)
            print(f"GROQ SUCCESS: {model_name}")
            return result, f"groq:{model_name}"
        except Exception as error:
            error_text = str(error)
            errors.append(f"{model_name}: {error_text}")
            print(f"GROQ FAILED ({model_name}): {error_text[:120]}")
            continue

    raise RuntimeError("All Groq models failed:\n" + "\n".join(errors))


# -------------------------------------------------------------
# 3. OPENROUTER PROVIDER
# -------------------------------------------------------------

def caption_with_openrouter_model(audio_path, model_name):
    if openrouter_client is None:
        raise RuntimeError("OpenRouter client not initialized.")

    print(f"[OPENROUTER] Trying {model_name}")
    features = analyze_audio_locally(audio_path)
    prompt = f"{ANALYSIS_INSTRUCTION}\n\nObjective acoustic measurements:\n{json.dumps(features, indent=2)}"

    def call_api():
        completion = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
        )
        return completion.choices[0].message.content

    raw_text = execute_with_retry(call_api, "openrouter", model_name)
    result = clean_json_response(raw_text)
    result = normalize_result(result)

    if result["tempo_bpm"] is None:
        result["tempo_bpm"] = features["estimated_tempo_bpm"]

    return result


def caption_with_openrouter(audio_path):
    if openrouter_client is None:
        raise RuntimeError("OpenRouter client not initialized.")

    errors = []
    for model_name in OPENROUTER_MODELS:
        if not is_model_available("openrouter", model_name):
            continue
        try:
            result = caption_with_openrouter_model(audio_path, model_name)
            print(f"OPENROUTER SUCCESS: {model_name}")
            return result, f"openrouter:{model_name}"
        except Exception as error:
            error_text = str(error)
            errors.append(f"{model_name}: {error_text}")
            print(f"OPENROUTER FAILED ({model_name}): {error_text[:120]}")
            continue

    raise RuntimeError("All OpenRouter models failed:\n" + "\n".join(errors))


# -------------------------------------------------------------
# 4. OPENAI PROVIDER
# -------------------------------------------------------------

def caption_with_openai(audio_path):
    if openai_client is None or not OPENAI_ENABLED:
        raise RuntimeError("OpenAI client not enabled.")

    if not is_model_available("openai", OPENAI_MODEL):
        raise RuntimeError("OpenAI model on cooldown or disabled.")

    print(f"[OPENAI] Trying {OPENAI_MODEL} on {audio_path.name}")
    features = analyze_audio_locally(audio_path)
    prompt = f"{ANALYSIS_INSTRUCTION}\n\nObjective acoustic measurements:\n{json.dumps(features, indent=2)}"

    def call_api():
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content

    raw_text = execute_with_retry(call_api, "openai", OPENAI_MODEL)
    result = clean_json_response(raw_text)
    result = normalize_result(result)

    if result["tempo_bpm"] is None:
        result["tempo_bpm"] = features["estimated_tempo_bpm"]

    return result


# -------------------------------------------------------------
# 5. DETERMINISTIC LOCAL ACOUSTIC FALLBACK ENGINE
# -------------------------------------------------------------

def caption_with_local_features(audio_path):
    """
    Deterministic acoustic rule-based fallback engine.
    Ensures 100% caption generation fallback with zero API dependencies.
    """
    print(f"[LOCAL_FALLBACK] Synthesizing acoustic caption for {audio_path.name}")
    features = analyze_audio_locally(audio_path)

    bpm = features["estimated_tempo_bpm"]
    centroid = features["spectral_centroid_hz"]
    zcr = features["zero_crossing_rate"]
    low_e = features["low_frequency_energy"]
    mid_e = features["mid_frequency_energy"]
    high_e = features["high_frequency_energy"]

    # Infer tempo description
    if bpm < 85:
        tempo_desc = "slow, relaxed tempo"
    elif bpm < 115:
        tempo_desc = "mid-tempo groove"
    elif bpm < 135:
        tempo_desc = "upbeat driving tempo"
    else:
        tempo_desc = "fast energetic tempo"

    # Infer instruments from spectral energy
    instruments = []
    if low_e > 0.04:
        instruments.extend(["deep bass synth", "punchy kick drum"])
    else:
        instruments.append("subdued bass")

    if mid_e > 0.04:
        instruments.extend(["melodic synthesizer", "rhythmic chords"])
    else:
        instruments.append("ambient synth pad")

    if high_e > 0.015 or zcr > 0.04:
        instruments.extend(["hi-hats", "crisp percussion"])

    # Infer mood & style
    moods = []
    if centroid > 2800:
        moods.extend(["bright", "energetic"])
        style = "Modern Electronic / Dance"
        genre = "Electronic"
    elif centroid > 1600:
        moods.extend(["melodic", "groovy"])
        style = "Synthwave / Pop"
        genre = "Pop"
    else:
        moods.extend(["atmospheric", "mellow", "ambient"])
        style = "Ambient / Chillout"
        genre = "Ambient"

    rhythm = f"{tempo_desc} at {round(bpm)} BPM with steady beat pattern"
    structure = "consistent 30-second musical section with balanced dynamic arrangement"
    full_desc = f"Instrumental {genre.lower()} track featuring {', '.join(instruments[:3])}, characterized by {rhythm} and a {moods[0]} vibe."
    musicgen_prompt = f"{genre.lower()} track, {round(bpm)} bpm, {', '.join(instruments[:3])}, {', '.join(moods[:2])}, {rhythm}"

    result = {
        "genre": genre,
        "style": style,
        "mood": moods,
        "instruments": instruments,
        "tempo_bpm": round(bpm, 1),
        "rhythm": rhythm,
        "structure": structure,
        "full_description": full_desc,
        "musicgen_prompt": musicgen_prompt,
    }

    return result, "local:acoustic_analyzer"


# =============================================================
# CSV PERSISTENCE & HELPERS
# =============================================================

def load_existing_results(output_csv):
    output_csv = Path(output_csv)
    if not output_csv.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_csv)
    except Exception:
        return pd.DataFrame()


def save_results(df, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def build_result_row(metadata, result, source):
    return {
        "chunk_id": metadata["chunk_id"],
        "track_id": metadata["track_id"],
        "split": metadata["split"],
        "audio_path": metadata["audio_path"],
        "status": "success",
        "caption_source": source,
        "genre": result.get("genre", ""),
        "style": result.get("style", ""),
        "mood": json.dumps(result.get("mood", []), ensure_ascii=False),
        "instruments": json.dumps(result.get("instruments", []), ensure_ascii=False),
        "tempo_bpm": result.get("tempo_bpm", None),
        "rhythm": result.get("rhythm", ""),
        "structure": result.get("structure", ""),
        "full_description": result.get("full_description", ""),
        "musicgen_prompt": result.get("musicgen_prompt", ""),
        "error": "",
    }


def build_failed_row(metadata, error):
    return {
        "chunk_id": metadata["chunk_id"],
        "track_id": metadata["track_id"],
        "split": metadata["split"],
        "audio_path": metadata["audio_path"],
        "status": "failed",
        "caption_source": "",
        "genre": "",
        "style": "",
        "mood": "",
        "instruments": "",
        "tempo_bpm": None,
        "rhythm": "",
        "structure": "",
        "full_description": "",
        "musicgen_prompt": "",
        "error": str(error),
    }


def upsert_row(df, new_row, chunk_id):
    if not df.empty and "chunk_id" in df.columns:
        df = df[df["chunk_id"].astype(str) != str(chunk_id)]

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


# =============================================================
# MAIN PIPELINE
# =============================================================

def run_caption_pipeline(
    chunks_csv="data/metadata/chunks.csv",
    output_csv="data/metadata/chunk_captions.csv",
    project_root=".",
):
    chunks_csv = Path(chunks_csv)
    output_csv = Path(output_csv)
    project_root = Path(project_root).resolve()

    if not chunks_csv.exists():
        raise FileNotFoundError(f"chunks.csv not found: {chunks_csv}")

    chunks_df = pd.read_csv(chunks_csv)
    if chunks_df.empty:
        raise ValueError("chunks.csv is empty.")

    print("\n" + "=" * 75)
    print("MULTI-PROVIDER RESILIENT MUSIC CAPTIONING")
    print("=" * 75)

    print("PRIMARY TIER   : Gemini (Multimodal Native Audio / Text Fallback)")
    print("SECONDARY TIER : Groq (Vision Spectrogram / Text Models)")
    print("TERTIARY TIER  : OpenRouter (Multi-model Free Pool)")
    print("FINAL SAFETY   : Deterministic Local Acoustic Analyzer (100% Success Guarantee)")

    df = load_existing_results(output_csv)
    completed_chunks = set()

    if not df.empty and "chunk_id" in df.columns and "status" in df.columns:
        successful = df[df["status"].astype(str).str.lower().eq("success")]
        completed_chunks = set(successful["chunk_id"].astype(str))

    print("\n" + "=" * 75)
    print("RESUMABLE PIPELINE STATUS")
    print("=" * 75)
    print(f"Total chunks in dataset : {len(chunks_df)}")
    print(f"Already completed      : {len(completed_chunks)}")
    print(f"Remaining to process   : {len(chunks_df) - len(completed_chunks)}")

    for index, row in chunks_df.iterrows():
        chunk_id = str(row["chunk_id"])

        print("\n" + "-" * 75)
        print(f"Progress: {index + 1}/{len(chunks_df)} | Chunk: {chunk_id}")

        if chunk_id in completed_chunks:
            print("SKIP: Already successful in CSV")
            continue

        audio_path = resolve_audio_path(row["audio_path"], project_root)
        print(f"Audio: {audio_path}")

        if not audio_path.exists():
            print(f"ERROR: File not found {audio_path}")
            failed_row = build_failed_row(row, f"File not found: {audio_path}")
            df = upsert_row(df, failed_row, chunk_id)
            save_results(df, output_csv)
            continue

        result = None
        source = ""
        errors = []

        # -----------------------------------------------------
        # 1. GEMINI (PRIMARY)
        # -----------------------------------------------------
        if gemini_client and result is None:
            try:
                result, source = caption_with_gemini(audio_path)
            except Exception as exc:
                errors.append(f"Gemini: {exc}")

        # -----------------------------------------------------
        # 2. GROQ (SECONDARY)
        # -----------------------------------------------------
        if groq_client and result is None:
            try:
                result, source = caption_with_groq(audio_path)
            except Exception as exc:
                errors.append(f"Groq: {exc}")

        # -----------------------------------------------------
        # 3. OPENROUTER (TERTIARY)
        # -----------------------------------------------------
        if openrouter_client and result is None:
            try:
                result, source = caption_with_openrouter(audio_path)
            except Exception as exc:
                errors.append(f"OpenRouter: {exc}")

        # -----------------------------------------------------
        # 4. OPENAI (OPTIONAL)
        # -----------------------------------------------------
        if openai_client and OPENAI_ENABLED and result is None:
            try:
                result = caption_with_openai(audio_path)
                source = f"openai:{OPENAI_MODEL}"
            except Exception as exc:
                errors.append(f"OpenAI: {exc}")

        # -----------------------------------------------------
        # 5. DETERMINISTIC LOCAL FALLBACK (GUARANTEED SUCCESS)
        # -----------------------------------------------------
        if result is None:
            print(f"[FALLBACK TRIGGERED] All online APIs temporarily unavailable/exhausted.")
            print(f"API Errors encountered:\n  " + "\n  ".join(errors))
            result, source = caption_with_local_features(audio_path)

        # -----------------------------------------------------
        # SAVE SUCCESSFUL RESULT
        # -----------------------------------------------------
        new_row = build_result_row(row, result, source)
        df = upsert_row(df, new_row, chunk_id)
        save_results(df, output_csv)
        completed_chunks.add(chunk_id)

        print("CAPTION SUCCESS")
        print(f"Source: {source}")
        print(f"MusicGen prompt: {result['musicgen_prompt']}")

        time.sleep(DELAY_SECONDS)

    # =========================================================
    # SUMMARY
    # =========================================================
    success_count = int(df["status"].astype(str).str.lower().eq("success").sum()) if not df.empty and "status" in df.columns else 0
    failed_count = int(df["status"].astype(str).str.lower().eq("failed").sum()) if not df.empty and "status" in df.columns else 0

    print("\n" + "=" * 75)
    print("CAPTIONING COMPLETED")
    print("=" * 75)
    print(f"Successful: {success_count}")
    print(f"Failed    : {failed_count}")

    if not df.empty and "caption_source" in df.columns:
        print("\nCaption sources breakdown:")
        print(df[df["status"].astype(str).str.lower().eq("success")]["caption_source"].value_counts())

    print(f"\nOutput file: {output_csv}")
    return df


if __name__ == "__main__":
    run_caption_pipeline()