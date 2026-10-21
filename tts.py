import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device != "cpu" else torch.float32
print(f"Usando: {device}")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map=device,
    dtype=dtype,
)

wavs, sr = model.generate_custom_voice(
    text="Hola, este es un taller de Inteligencia Artificial Generativa usando Qwen TTS.",
    language="Spanish",
    speaker="Ryan",
)
sf.write("salida_qwen.wav", wavs[0], sr)
print("Audio generado: salida_qwen.wav")
