import os
import tempfile

from flask import Flask, Response, jsonify, request
from melo.api import TTS

# 메인 프로젝트(.venv)와 분리된 .venv-tts에서 실행되는 독립 프로세스.
# src/tts/tts_client.py가 HTTP로 이 서버를 호출한다 (Ollama와 동일한 패턴).

app = Flask(__name__)

_MODEL = TTS(language="KR", device="cpu")
_SPEAKER_ID = _MODEL.hps.data.spk2id["KR"]


@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _MODEL.tts_to_file(text, _SPEAKER_ID, path, speed=1.0)
        with open(path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.remove(path)

    return Response(audio_bytes, mimetype="audio/wav")


if __name__ == "__main__":
    # 0.0.0.0으로 바인딩해야 Docker 컨테이너 밖(-p 포트 매핑)에서 접근 가능하다.
    app.run(host="0.0.0.0", port=8890)
