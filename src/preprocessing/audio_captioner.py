import os
import json
import time
import base64
import tempfile
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display 
from pathlib import Path
from google import genai
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set."
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


GEMINI_MODEL = "gemini-2.5-flash"

# Current Groq multimodal model
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"


RAW_DIR = Path("data/raw")
OUTPUT_CSV = Path("data/metadata/captions.csv")

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
}


# ============================================================
# COMMON PROMPT
# ============================================================

CAPTION_PROMPT = """
Analyze this music/audio carefully.

Return ONLY valid JSON using exactly this structure:

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

- genre: main musical genre
- style: musical style
- mood: list of moods
- instruments: only instruments that can reasonably be identified
- tempo_bpm: estimated BPM if possible, otherwise null
- rhythm: describe rhythmic characteristics
- structure: describe how the music develops
- full_description: detailed but factual description
- musicgen_prompt: concise 1–3 sentence prompt suitable for
  generating similar INSTRUMENTAL music with MusicGen

Do not invent instruments.
If an instrument or musical property is uncertain, leave it
out or state uncertainty in the description.

Do not include markdown.
Return JSON only.
"""


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json_response(text):
    """Remove accidental markdown fences and parse JSON."""

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return json.loads(text.strip())


# ============================================================
# GEMINI CAPTIONER
# ============================================================

def caption_with_gemini(audio_path):
    """
    Primary captioning method.

    Gemini receives the actual audio file.
    """

    print(f"\n[GEMINI] Uploading {audio_path.name}")

    uploaded_file = gemini_client.files.upload(
        file=str(audio_path)
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            uploaded_file,
            CAPTION_PROMPT
        ]
    )

    return clean_json_response(response.text)


# ============================================================
# LOCAL AUDIO ANALYSIS
# ============================================================

def analyze_audio_locally(audio_path):
    """
    Extract objective audio features locally.

    These features are passed to Groq along with a spectrogram.
    """

    y, sr = librosa.load(
        str(audio_path),
        sr=None,
        mono=True
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    rms = float(
        librosa.feature.rms(y=y).mean()
    )

    spectral_centroid = float(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        ).mean()
    )

    spectral_bandwidth = float(
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        ).mean()
    )

    zero_crossing_rate = float(
        librosa.feature.zero_crossing_rate(y).mean()
    )

    # Estimate tempo
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    # librosa versions can return numpy arrays
    try:
        tempo = float(tempo[0])
    except (TypeError, IndexError):
        tempo = float(tempo)

    return {
        "sample_rate": int(sr),
        "duration_seconds": round(duration, 2),
        "rms": round(rms, 6),
        "spectral_centroid_hz": round(
            spectral_centroid,
            2
        ),
        "spectral_bandwidth_hz": round(
            spectral_bandwidth,
            2
        ),
        "zero_crossing_rate": round(
            zero_crossing_rate,
            6
        ),
        "estimated_tempo_bpm": round(
            tempo,
            2
        )
    }


# ============================================================
# CREATE SPECTROGRAM
# ============================================================

def create_spectrogram(audio_path):
    """
    Create a mel spectrogram image for Groq vision analysis.
    """

    y, sr = librosa.load(
        str(audio_path),
        sr=None,
        mono=True
    )

    # Limit analysis to the first 60 seconds.
    # This keeps the image representative and lightweight.
    max_samples = int(sr * 60)

    y = y[:max_samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    temp_path = Path(temp_file.name)
    temp_file.close()

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        f"Mel Spectrogram - {audio_path.name}"
    )

    plt.tight_layout()

    plt.savefig(
        temp_path,
        dpi=120
    )

    plt.close()

    return temp_path


# ============================================================
# ENCODE IMAGE
# ============================================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ============================================================
# GROQ FALLBACK
# ============================================================

def caption_with_groq(audio_path):
    """
    Fallback captioning.

    Groq's documented audio endpoint is speech-to-text,
    not direct music captioning.

    Therefore we provide Groq with:
      1. A mel spectrogram image
      2. Objective local audio features

    Groq then produces structured musical metadata.
    """

    print(
        f"\n[GROQ FALLBACK] Analyzing "
        f"{audio_path.name}"
    )

    # ---------------------------------------------
    # Local analysis
    # ---------------------------------------------

    features = analyze_audio_locally(
        audio_path
    )

    print("\nLocal audio features:")
    print(
        json.dumps(
            features,
            indent=2
        )
    )

    # ---------------------------------------------
    # Spectrogram
    # ---------------------------------------------

    spectrogram_path = create_spectrogram(
        audio_path
    )

    try:

        base64_image = encode_image(
            spectrogram_path
        )

        groq_prompt = f"""
You are analyzing an instrumental music recording.

The attached image is a MEL SPECTROGRAM of the audio.

Use BOTH:
1. the spectrogram
2. the objective audio measurements below

Audio measurements:

{json.dumps(features, indent=2)}

{CAPTION_PROMPT}

Important:

- The spectrogram does NOT prove the exact instrument.
- Do not invent specific instruments.
- If instruments cannot be confidently identified,
  return an empty instruments list.
- Tempo can use the provided estimated tempo as evidence,
  but treat it as an estimate.
- Focus on genre/style/mood/rhythm/texture/structure.
- This is instrumental music unless there is clear evidence
  of vocals.
"""

        completion = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": groq_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                    f"data:image/png;base64,"
                                    f"{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500,
            response_format={
                "type": "json_object"
            }
        )

        text = (
            completion
            .choices[0]
            .message
            .content
        )

        result = clean_json_response(
            text
        )

        # If Groq did not return tempo,
        # use our local estimate.
        if result.get("tempo_bpm") is None:
            result["tempo_bpm"] = features[
                "estimated_tempo_bpm"
            ]

        return result

    finally:

        # Delete temporary spectrogram
        try:
            spectrogram_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if OUTPUT_CSV.exists():

        print(
            "\nExisting captions.csv found."
        )

        df = pd.read_csv(
            OUTPUT_CSV
        )

        print(
            f"Existing rows: {len(df)}"
        )

        return df

    print(
        "\nNo existing captions.csv found."
    )

    return pd.DataFrame()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nSaved progress → {OUTPUT_CSV}"
    )


# ============================================================
# BUILD RESULT ROW
# ============================================================

def build_result_row(
    filename,
    result,
    source
):

    return {
        "filename": filename,

        "status": "success",

        "caption_source": source,

        "genre": result.get(
            "genre",
            ""
        ),

        "style": result.get(
            "style",
            ""
        ),

        "mood": json.dumps(
            result.get(
                "mood",
                []
            ),
            ensure_ascii=False
        ),

        "instruments": json.dumps(
            result.get(
                "instruments",
                []
            ),
            ensure_ascii=False
        ),

        "tempo_bpm": result.get(
            "tempo_bpm",
            None
        ),

        "rhythm": result.get(
            "rhythm",
            ""
        ),

        "structure": result.get(
            "structure",
            ""
        ),

        "full_description": result.get(
            "full_description",
            ""
        ),

        "musicgen_prompt": result.get(
            "musicgen_prompt",
            ""
        ),

        "error": ""
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print(
        "RESUMABLE AUDIO CAPTIONING PIPELINE"
    )
    print(
        "Gemini → Groq Vision Fallback"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Find audio files
    # --------------------------------------------------------

    audio_files = sorted(
        [
            p
            for p in RAW_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    print(
        f"\nFound {len(audio_files)} audio files."
    )

    # --------------------------------------------------------
    # Existing CSV
    # --------------------------------------------------------

    df = load_existing_results()

    # --------------------------------------------------------
    # Determine successful files
    # --------------------------------------------------------

    completed_files = set()

    if (
        not df.empty
        and "filename" in df.columns
        and "status" in df.columns
    ):

        successful_rows = df[
            df["status"]
            .astype(str)
            .str.lower()
            .eq("success")
        ]

        completed_files = set(
            successful_rows[
                "filename"
            ].astype(str)
        )

    print(
        f"\nAlready completed: "
        f"{len(completed_files)}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, audio_path in enumerate(
        audio_files,
        start=1
    ):

        filename = audio_path.name

        print("\n" + "-" * 65)

        print(
            f"{index}/{len(audio_files)} "
            f"{filename}"
        )

        # ====================================================
        # SKIP SUCCESSFUL
        # ====================================================

        if filename in completed_files:

            print(
                "STATUS: Already successful"
            )

            print(
                "ACTION: SKIP"
            )

            continue

        # ====================================================
        # PRIMARY: GEMINI
        # ====================================================

        result = None
        source = None
        gemini_error = None

        try:

            print(
                "\nTrying Gemini..."
            )

            result = caption_with_gemini(
                audio_path
            )

            source = "gemini"

            print(
                "\nGemini SUCCESS"
            )

        except Exception as e:

            gemini_error = str(e)

            print(
                "\nGemini FAILED:"
            )

            print(
                gemini_error
            )

        # ====================================================
        # FALLBACK: GROQ
        # ====================================================

        if result is None:

            print(
                "\nSwitching to Groq fallback..."
            )

            try:

                result = caption_with_groq(
                    audio_path
                )

                source = "groq"

                print(
                    "\nGroq FALLBACK SUCCESS"
                )

            except Exception as e:

                groq_error = str(e)

                print(
                    "\nGroq FALLBACK FAILED:"
                )

                print(
                    groq_error
                )

                # ------------------------------------------
                # Save failed result
                # ------------------------------------------

                failed_row = {
                    "filename": filename,

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

                    "error": (
                        f"Gemini: {gemini_error} | "
                        f"Groq: {groq_error}"
                    )
                }

                if (
                    not df.empty
                    and "filename" in df.columns
                ):

                    df = df[
                        df["filename"]
                        .astype(str)
                        != filename
                    ]

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [failed_row]
                        )
                    ],
                    ignore_index=True
                )

                save_results(df)

                continue

        # ====================================================
        # SUCCESS
        # ====================================================

        new_row = build_result_row(
            filename,
            result,
            source
        )

        # Remove old failed row
        if (
            not df.empty
            and "filename" in df.columns
        ):

            df = df[
                df["filename"]
                .astype(str)
                != filename
            ]

        # Add successful result
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [new_row]
                )
            ],
            ignore_index=True
        )

        # Save immediately
        save_results(df)

        completed_files.add(
            filename
        )

        print(
            "\nCaption source:"
            f" {source}"
        )

        print(
            "\nMusicGen prompt:"
        )

        print(
            result.get(
                "musicgen_prompt",
                ""
            )
        )

        # Small delay
        time.sleep(2)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 65)

    if (
        not df.empty
        and "status" in df.columns
    ):

        success_count = (
            df["status"]
            .astype(str)
            .str.lower()
            .eq("success")
            .sum()
        )

        failed_count = (
            df["status"]
            .astype(str)
            .str.lower()
            .eq("failed")
            .sum()
        )

        print(
            f"Successful: {success_count}"
        )

        print(
            f"Failed: {failed_count}"
        )

        if "caption_source" in df.columns:

            print("\nCaption sources:")

            print(
                df[
                    df["status"]
                    .astype(str)
                    .str.lower()
                    .eq("success")
                ][
                    "caption_source"
                ].value_counts()
            )

    print(
        f"\nOutput: {OUTPUT_CSV}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
