import os
import whisper
import pyaudio
import numpy as np
import asyncio
from datetime import datetime

# 初始化 Whisper 模型
model = whisper.load_model("turbo")

# 配置音频流
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始实时语音转写...")

async def transcribe_audio(frames):
    # print("===开始音频转写===")
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    result = model.transcribe(audio_data, language="zh", fp16=False)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}]：{result['text']}")

async def main():
    frames = []
    try:
        while True:
            # print("===读取音频数据===")
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except OSError as e:
                    print(f"Error reading audio data: {e}")
            # print("===创建转写任务===")
            asyncio.get_event_loop().create_task(transcribe_audio(frames))
            frames = []
            await asyncio.sleep(0)  # 让出控制权，确保任务被调度
    except KeyboardInterrupt:
        print("停止实时语音转写")
    finally:
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()

# 运行异步主函数
asyncio.run(main())