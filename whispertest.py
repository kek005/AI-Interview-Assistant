import time
import torch
import numba
from faster_whisper import WhisperModel

# 🚀 **Enable Maximum GPU Performance**
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat32 for speedup
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # FP16 optimization
torch.set_float32_matmul_precision('high')  # Maximize efficiency
torch.backends.cudnn.benchmark = True  # Enable cuDNN tuning for faster inference

# 🔥 **Check GPU Availability**
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"🔥 GPU is enabled: {device_name}")
else:
    raise RuntimeError("⚠️ GPU is NOT available! Check CUDA installation.")

# 🚀 **Preload Whisper Model into VRAM (Persistent)**
print("🔥 Loading Whisper Model into VRAM...")
model = WhisperModel("tiny", device="cuda", compute_type="float16")

# ✅ **Numba-Optimized Audio Preprocessing**
@numba.jit(nopython=True, cache=True)
def fast_preprocess(audio):
    """ Optimized audio preprocessing (if needed) """
    return audio  # Placeholder for future optimizations

def transcribe_audio(file):
    """ Transcribes an audio file with minimal overhead """
    start_time = time.perf_counter()

    # 🔥 **Zero-Overhead Transcription**
    audio = fast_preprocess(file)
    segments, _ = model.transcribe(audio, beam_size=1, vad_filter=True)

    end_time = time.perf_counter()
    print(f"✅ Transcription completed in: {(end_time - start_time) * 1000:.2f} ms")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# 🏆 **Zero-Lag Transcription**
transcribe_audio("test_audio.mp3")