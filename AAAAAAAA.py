import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

AUDIO_PATH = "/home/cansu/Downloads/Voices/GLaDOS/00_part1_entry-2.wav"

model = ChatterboxTTS.from_pretrained(device)
wav = model.generate(
    "Thank you for tuning to the yuri jukebox. Our next song is fourty thousand yyears by DHigh On Fire. We hope you enjoy.",
    audio_prompt_path=AUDIO_PATH,
)
ta.save("testvc.wav", wav, model.sr)
