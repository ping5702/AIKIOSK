import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline

TARGET_SAMPLE_RATE = 16000


class WhisperSTT:
    """Whisper 계열 모델(예: SungBeom/whisper-small-ko)로 오디오를 텍스트로 변환한다."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)

    def transcribe(self, audio_path: str) -> str:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        return self.transcribe_array(audio, sample_rate)

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

        result = self.pipe({"array": audio, "sampling_rate": TARGET_SAMPLE_RATE})
        return result["text"].strip()
