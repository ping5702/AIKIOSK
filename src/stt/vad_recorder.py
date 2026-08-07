import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
# Silero VAD는 16kHz 기준으로 정확히 512 샘플(32ms) 단위 청크를 요구한다.
CHUNK_SIZE = 512
CHUNK_DURATION_MS = CHUNK_SIZE / SAMPLE_RATE * 1000
SILENCE_TIMEOUT_MS = 1000
SILENCE_CHUNKS = round(SILENCE_TIMEOUT_MS / CHUNK_DURATION_MS)
SPEECH_THRESHOLD = 0.5


class VadRecorder:
    """마이크 입력에서 Silero VAD(신경망 기반)로 발화 구간(말 시작 ~ 1초 무음)만
    자동으로 녹음한다. webrtcvad(에너지 기반)보다 노이즈 환경에서 더 정확하다."""

    def __init__(self, threshold: float = SPEECH_THRESHOLD):
        self.model = load_silero_vad()
        self.threshold = threshold

    def _is_speech(self, chunk_int16: np.ndarray) -> bool:
        chunk_float = chunk_int16.astype(np.float32) / 32768.0
        with torch.no_grad():
            prob = self.model(torch.from_numpy(chunk_float), SAMPLE_RATE).item()
        return prob >= self.threshold

    def record_utterance(self, max_seconds: float = 15.0) -> np.ndarray:
        """발화 하나를 녹음해 16kHz float32 numpy 배열([-1, 1])로 반환한다.
        음성이 감지되지 않으면 빈 배열을 반환한다."""
        max_chunks = int(max_seconds * 1000 / CHUNK_DURATION_MS)
        frames = []
        triggered = False
        silence_count = 0

        self.model.reset_states()
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(CHUNK_SIZE)
                chunk = chunk.flatten()
                is_speech = self._is_speech(chunk)

                if not triggered:
                    if is_speech:
                        triggered = True
                        frames.append(chunk)
                else:
                    frames.append(chunk)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS:
                            break

        if not frames:
            return np.array([], dtype=np.float32)

        audio_int16 = np.concatenate(frames)
        return audio_int16.astype(np.float32) / 32768.0
