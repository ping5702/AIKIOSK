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

    yield history, "인식 중...", None
    text = stt.transcribe_array(audio, 16000)
    history = history + [{"role": "user", "content": text}]
    yield history, "답변 생성 중...", None

    answer = pipeline.answer(text)
    history = history + [{"role": "assistant", "content": answer}]
    yield history, "음성 합성 중...", None

    audio_path = pipeline.synthesize_to_file(answer)
    status = "완료" if audio_path else "TTS 서버에 연결할 수 없어 텍스트만 표시합니다."
    yield history, status, (str(audio_path) if audio_path else None)


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
    demo.launch()
