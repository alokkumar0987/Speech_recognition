import whisper
import torch
import math

# Load model
model = whisper.load_model("tiny")  # choose tiny/base/small/medium/large

# Load audio
audio_path = "listening.wav"
audio = whisper.load_audio(audio_path)

# For long audio, split into chunks of ~30 seconds
sample_rate = whisper.audio.SAMPLE_RATE  # 16000
max_len = 30 * sample_rate  # 30 seconds in samples
num_chunks = math.ceil(len(audio) / max_len)

transcription = ""
for i in range(num_chunks):
    start = i * max_len
    end = min((i + 1) * max_len, len(audio))
    chunk = audio[start:end]

    if len(chunk) < 100:  # skip very short chunks (optional)
        continue

    # Pad or trim to exactly 30s before spectrogram
    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # Detect language (only once, on the first chunk)
    if i == 0:
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    transcription += result.text + " "

print("Full transcription:")
print(transcription.strip())
