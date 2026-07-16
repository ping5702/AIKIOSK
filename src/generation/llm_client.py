import requests


class OllamaClient:
    """로컬 Ollama 서버(localhost:11434)에 프롬프트를 보내 텍스트를 생성하는 클라이언트."""

    def __init__(self, model: str, host: str, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
