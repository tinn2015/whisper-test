import whisper

def transcribe_audio(file_path):
    model = whisper.load_model("turbo")
    result = model.transcribe(file_path, language="zh",prompt="这是一个句子。另一个句子！注意添加标点符号。")
    return result['text']

if __name__ == "__main__":
    audio_file = "test7.mp4"
    transcription = transcribe_audio(audio_file)
    print(transcription)