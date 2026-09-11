import gradio as gr

import config
from src.kiosk_pipeline import KioskPipeline
from src.stt.vad_recorder import VadRecorder
from src.stt.whisper_stt import WhisperSTT

pipeline = KioskPipeline()
recorder = VadRecorder()
stt = WhisperSTT(model_name=config.STT_MODEL_NAME, device=config.STT_DEVICE)


def speak_turn(history):
    history = history or []
    yield history, "듣는 중... (말이 끝나고 1초간 조용하면 자동으로 멈춥니다)", None

    audio = recorder.record_utterance()
    if audio.size == 0:
        yield history, "음성이 감지되지 않았습니다. 다시 눌러주세요.", None
        return

    # STT(수 초 소요)를 시작하기 전에 대기음부터 재생해야, 사용자가 말을 마친
    # 직후 바로 반응이 온다. STT+RAG+LLM+TTS 전체가 끝날 때까지 반복 재생한다.
    waiting_audio = str(config.WAITING_AUDIO_PATH) if config.WAITING_AUDIO_PATH.exists() else None
    yield history, "인식 중...", gr.Audio(value=waiting_audio, autoplay=True, loop=True)

    text = stt.transcribe_array(audio, 16000)
    corrected = pipeline.correct_query(text)
    display_text = text if corrected == text else f"{text} → (보정됨) {corrected}"
    history = history + [{"role": "user", "content": display_text}]
    yield history, "답변 생성 중...", gr.Audio(value=waiting_audio, autoplay=True, loop=True)

    answer = pipeline.answer(corrected)
    history = history + [{"role": "assistant", "content": answer}]
    yield history, "음성 합성 중...", gr.Audio(value=waiting_audio, autoplay=True, loop=True)

    audio_path = pipeline.synthesize_to_file(answer)
    status = "완료" if audio_path else "TTS 서버에 연결할 수 없어 텍스트만 표시합니다 (TTS 컨테이너가 켜져 있는지 확인하세요)."
    final_audio = gr.Audio(value=str(audio_path) if audio_path else None, autoplay=True, loop=False)
    yield history, status, final_audio


with gr.Blocks(title="AI 안내 키오스크") as demo:
    gr.Markdown("# AI 안내 키오스크")
    chatbot = gr.Chatbot(label="대화 내역")
    status = gr.Textbox(label="상태", interactive=False)
    audio_player = gr.Audio(label="답변 음성", autoplay=True)
    speak_button = gr.Button("🎙️ 말하기", variant="primary")

    speak_button.click(
        fn=speak_turn,
        inputs=[chatbot],
        outputs=[chatbot, status, audio_player],
    )

if __name__ == "__main__":
    demo.launch(footer_links=[])
