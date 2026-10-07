import os
import sys
from pathlib import Path

# Monkey-patch gradio_client schema parser to prevent Gradio bug with boolean schemas in get_api_info
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def safe_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = safe_get_type

    _orig_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type
    def safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _orig_json_schema_to_python_type(schema, defs)
    gradio_client.utils._json_schema_to_python_type = safe_json_schema_to_python_type
except Exception as patch_err:
    print(f"Warning patching gradio_client: {patch_err}")

import gradio as gr

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False
    class DummySpaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                return lambda f: f
            return func
    spaces = DummySpaces()

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.music_generator import MusicGenerator

print("Loading MusicGenerator model for Hugging Face Space...")
generator = MusicGenerator(
    base_model_name="facebook/musicgen-small",
    adapter_path="models/adapters/adapter3",
)

try:
    generator.load_model()
except Exception as e:
    print(f"Startup warning loading model: {e}")


@spaces.GPU(duration=120)
def generate_music(prompt, max_new_tokens, temperature, guidance_scale, top_k):
    """Gradio handler function for music generation."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a valid music prompt.")

    try:
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            guidance_scale=float(guidance_scale),
            top_k=int(top_k),
            output_dir="outputs",
        )
        return result["file"]
    except Exception as exc:
        raise gr.Error(f"Generation error: {str(exc)}")


# Build Gradio UI
with gr.Blocks(title="Indian Music Generator (MusicGen + LoRA)") as demo:
    gr.Markdown(
        """
        # 🎵 Indian Music Generator
        Fine-tuned **`facebook/musicgen-small`** with custom **LoRA Adapter** for Indian classical & instrumental music generation.
        """
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Music Prompt",
                placeholder="e.g. Indian classical sitar with tabla and tanpura drone",
                value="Indian classical sitar with tabla and tanpura drone",
                lines=3,
            )

            with gr.Accordion("Advanced Generation Settings", open=False):
                max_new_tokens_slider = gr.Slider(
                    minimum=250,
                    maximum=3000,
                    value=1500,
                    step=250,
                    label="Audio Length / Tokens (1500 ~ 30s)",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                )
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale",
                )
                top_k_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=250,
                    step=50,
                    label="Top-K Sampling",
                )

            generate_btn = gr.Button("🎶 Generate Music", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(
                label="Generated WAV Audio",
                type="filepath",
            )

    generate_btn.click(
        fn=generate_music,
        inputs=[
            prompt_input,
            max_new_tokens_slider,
            temperature_slider,
            guidance_scale_slider,
            top_k_slider,
        ],
        outputs=[audio_output],
    )

    gr.Examples(
        examples=[
            ["Indian classical sitar with tabla and tanpura drone, meditative raga melody"],
            ["Meditative bansuri bamboo flute instrumental, soft acoustic atmosphere"],
            ["Kashmiri santoor solo instrumental with fast rhythmic percussion"],
            ["Carnatic violin solo instrumental with mridangam beat"],
        ],
        inputs=[prompt_input],
    )

if __name__ == "__main__":
    demo.launch(show_api=False)

