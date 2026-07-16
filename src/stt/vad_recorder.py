import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_TIMEOUT_MS = 1000
SILENCE_FRAMES = SILENCE_TIMEOUT_MS // FRAME_DURATION_MS


class VadRecorder:
    """마이크 입력에서 webrtcvad로 발화 구간(말 시작 ~ 1초 무음)만 자동으로 녹음한다."""

    def __init__(self, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(aggressiveness)

    def record_utterance(self, max_seconds: float = 15.0) -> np.ndarray:
        """발화 하나를 녹음해 16kHz float32 numpy 배열([-1, 1])로 반환한다.
        음성이 감지되지 않으면 빈 배열을 반환한다."""
        max_frames = int(max_seconds * 1000 / FRAME_DURATION_MS)
        frames = []
        triggered = False
        silence_count = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SIZE,
        ) as stream:
            for _ in range(max_frames):
                frame, _ = stream.read(FRAME_SIZE)
                is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

                if not triggered:
                    if is_speech:
                        triggered = True
                        frames.append(frame)
                else:
                    frames.append(frame)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= SILENCE_FRAMES:
                            break

        if not frames:
            return np.array([], dtype=np.float32)

        audio_int16 = np.concatenate(frames).flatten()
        return audio_int16.astype(np.float32) / 32768.0
