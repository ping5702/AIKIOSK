import requests


class MeloTTSClient:
    """로컬 MeloTTS 서버(tts_server/server.py, 별도 .venv-tts에서 실행)를 호출하는 클라이언트."""

    def __init__(self, host: str, timeout: int = 60):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def synthesize(self, text: str) -> bytes:
        response = requests.post(
            f"{self.host}/synthesize",
            json={"text": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content
