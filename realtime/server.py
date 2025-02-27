import asyncio
import websockets
from faster_whisper import WhisperModel
import numpy as np
from datetime import datetime
import os
from scipy.io import wavfile
import tempfile
from pydub import AudioSegment
import io

# 初始化 Whisper 模型
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

async def transcribe_audio(audio_path):
    # 将WAV音频数据保存为临时文件
    import tempfile
    import sys
    
    # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
    #     temp_file.write(audio_data)
    #     temp_file_path = temp_file.name
    
    try:
        # 使用临时文件进行转写
        segments, _ = model.transcribe(audio_path, language="zh", beam_size=5, 
                                     vad_filter=True, 
                                     vad_parameters=dict(min_silence_duration_ms=500), 
                                     initial_prompt="这是中文普通话的句子")
        result_text = " ".join([segment.text for segment in segments])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 删除临时文件
        # os.unlink(temp_file_path)
        
        return f"[{current_time}]：{result_text}"
    except Exception as e:
        # 确保临时文件被删除
        # if os.path.exists(temp_file_path):
        #     os.unlink(temp_file_path)
        raise e

def generate_unique_filename(directory, prefix="record", extension=".wav"):
    counter = 1
    while True:
        if counter == 1:
            filename = f"{prefix}{extension}"
        else:
            filename = f"{prefix}_{counter}{extension}"
        
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

async def save_audio_data(audio_data, directory="."):
    filepath = generate_unique_filename(directory)
    with open(filepath, 'wb') as f:
        f.write(audio_data)
    return filepath

async def convert_webm_to_wav(webm_data):
    # 将WebM数据转换为WAV格式
    try:
        # 创建临时文件来存储WebM数据
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            temp_webm.write(webm_data)
            temp_webm_path = temp_webm.name

        # 使用pydub加载WebM文件并转换为WAV格式
        audio = AudioSegment.from_file(temp_webm_path, format='webm')
        
        # 确保音频采样率为16kHz（Whisper模型的标准要求）
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # 确保音频为单声道
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 将音频转换为字节数据
        wav_data = io.BytesIO()
        audio.export(wav_data, format='wav')
        
        # 删除临时WebM文件
        os.unlink(temp_webm_path)
        
        return wav_data.getvalue()
    except Exception as e:
        if os.path.exists(temp_webm_path):
            os.unlink(temp_webm_path)
        raise Exception(f"WebM转WAV失败: {str(e)}")

async def handle_websocket(websocket):
    print(f"新的客户端连接: {websocket.remote_address}")
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                print(f"接收到无效的消息类型: {type(message)}")
                continue

            # 检查音频数据长度
            if len(message) < 100:  # WebM文件头至少需要100字节
                print("音频数据太短或无效")
                continue

            try:
                # 转换WebM为WAV格式
                wav_data = await convert_webm_to_wav(message)
                
                # 保存转换后的WAV音频数据
                current_dir = os.path.dirname(os.path.abspath(__file__))
                saved_path = await save_audio_data(wav_data, current_dir)
                print(f"音频已保存到: {saved_path}")

                # 处理接收到的音频数据
                transcription = await transcribe_audio(saved_path)
                print(f"转写结果: {transcription}")

                # 发送转写结果回客户端
                await websocket.send(transcription)
                print(f"已发送转写结果给客户端: {websocket.remote_address}")

            except Exception as e:
                error_message = f"处理音频数据时出错: {str(e)}"
                print(error_message)
                try:
                    await websocket.send(f"错误: {error_message}")
                except:
                    pass  # 如果发送错误消息失败，忽略异常

    except websockets.exceptions.ConnectionClosed:
        print(f"客户端连接已关闭: {websocket.remote_address}")
    except Exception as e:
        print(f"WebSocket处理过程中出现错误: {str(e)}")
    finally:
        print(f"客户端连接已断开: {websocket.remote_address}")
        
async def main():
    server = await websockets.serve(
        handle_websocket,
        "localhost",
        8765,
        ping_interval=None  # 禁用ping以避免音频流传输中断
    )
    print("WebSocket服务器已启动，监听端口 8765...")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())