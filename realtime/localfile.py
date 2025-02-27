from faster_whisper import WhisperModel
import time
import os

def transcribe_local_audio(audio_path):
    """
    # 使用faster-whisper转写本地音频文件
    # 参数:
    #   audio_path: 音频文件路径
    #   model_size: 模型大小,可选"tiny","base","small","medium","large"
    # 返回:
    #   转写文本结果
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取音频文件完整路径
    audio_path = os.path.join(current_dir, audio_path)
    
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return None
    try:
        # 初始化模型
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        
        print(f"开始转写音频文件: {audio_path}")
        start_time = time.time()
        
        # 执行转写
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # 合并所有片段的文本
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
            
        end_time = time.time()
        print(f"转写完成! 用时: {end_time - start_time:.2f}秒，转写结果: {transcript}")
        
        return transcript.strip()
        
    except Exception as e:
        print(f"转写过程出错: {str(e)}")
        return None
transcribe_local_audio('./record.wav')
