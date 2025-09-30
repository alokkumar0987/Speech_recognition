import whisper

model=whisper.load_model("tiny")

result=model.transcribe("listening.wav")


print(result["text"])