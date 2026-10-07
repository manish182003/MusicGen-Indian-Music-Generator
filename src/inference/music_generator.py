import time
import uuid
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoProcessor, MusicgenForConditionalGeneration


class MusicGenerator:
    """
    MusicGenerator loads facebook/musicgen-small and fine-tuned PEFT LoRA adapter
    for generating audio WAV files from text prompts.
    """

    def __init__(
        self,
        base_model_name: str = "facebook/musicgen-small",
        adapter_path: str = "models/adapters/adapter3",
        device: str = None,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = None
        self.model = None

    def load_model(self):
        """Loads processor, base model, and PEFT LoRA adapter ONCE."""
        if self.model is not None:
            return

        print(f"Loading processor: {self.base_model_name}")
        self.processor = AutoProcessor.from_pretrained(self.base_model_name)

        print(f"Loading base model: {self.base_model_name}")
        base_model = MusicgenForConditionalGeneration.from_pretrained(
            self.base_model_name
        )

        if self.adapter_path.exists():
            print(f"Loading LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
            )
        else:
            print(f"Warning: Adapter path '{self.adapter_path}' not found. Using base model.")
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()
        print(f"MusicGenerator loaded successfully on device: {self.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1500,
        temperature: float = 1.0,
        guidance_scale: float = 3.5,
        top_k: int = 250,
        output_dir: str = "outputs",
    ) -> dict:
        """
        Generates audio for a given prompt and saves it as a WAV file.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if self.model is None:
            self.load_model()

        use_gpu = torch.cuda.is_available()
        target_device = "cuda" if use_gpu else "cpu"

        try:
            if use_gpu:
                self.model.to("cuda")

            inputs = self.processor(
                text=[prompt.strip()],
                padding=True,
                return_tensors="pt",
            ).to(target_device)

            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )

            sampling_rate = self.model.config.audio_encoder.sampling_rate

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            filename = f"generated_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
            file_path = out_path / filename

            waveform = audio_values[0, 0].cpu().numpy()
            sf.write(str(file_path), waveform, sampling_rate)

            return {
                "status": "completed",
                "file": str(file_path).replace("\\", "/"),
                "filename": filename,
                "sampling_rate": sampling_rate,
                "duration_seconds": round(len(waveform) / sampling_rate, 2),
            }
        finally:
            if use_gpu and self.model is not None:
                try:
                    self.model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
