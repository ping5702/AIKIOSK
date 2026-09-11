import soundfile as sf

import config
from src.kiosk_pipeline import KioskPipeline
from src.stt.vad_recorder import VadRecorder
from src.stt.whisper_stt import WhisperSTT

# 실제로 어떤 오디오가 녹음됐는지 확인하기 위한 디버그용 저장 경로
DEBUG_RECORDING_PATH = config.AUDIO_OUTPUT_DIR / "last_mic_recording.wav"


def main():
    recorder = VadRecorder()
    stt = WhisperSTT(model_name=config.STT_MODEL_NAME, device=config.STT_DEVICE)
    pipeline = KioskPipeline()

    print("마이크 키오스크 테스트 (Ctrl+C로 종료)")
    while True:
        input("\n말할 준비가 되면 Enter를 누르세요...")
        print("듣는 중... (말이 끝나고 1초간 조용하면 자동으로 멈춥니다)")
        audio = recorder.record_utterance()
        if audio.size == 0:
            print("(음성이 감지되지 않았습니다)")
            continue

        duration = len(audio) / 16000
        config.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(DEBUG_RECORDING_PATH, audio, 16000)
        print(f"(녹음 길이: {duration:.2f}초, 저장됨: {DEBUG_RECORDING_PATH})")

        text = stt.transcribe_array(audio, 16000)
        corrected = pipeline.correct_query(text)
        if corrected != text:
            print(f"[인식 결과] {text}  ->  [보정됨] {corrected}")
        else:
            print(f"[인식 결과] {text}")

        pipeline.answer_and_speak(corrected)


if __name__ == "__main__":
    main()
