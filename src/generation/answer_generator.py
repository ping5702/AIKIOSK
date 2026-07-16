from typing import Dict, List

from src.generation.llm_client import OllamaClient

_SYSTEM_PROMPT = (
    "너는 건물 안내 키오스크 도우미야. 아래 [검색된 장소 정보]에 있는 내용만 근거로 "
    "사용자 질문에 한국어로 친절하고 간결하게 답해. 정보에 없는 내용은 지어내지 말고, "
    "정보가 부족하면 '죄송합니다. 관련된 정보를 찾을 수 없습니다.'라고만 답해."
)


def _build_context(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        parts = [f"{i}. {r['name']} ({r['category']})", r["description"], f"위치: {r['location']}"]
        if r.get("hours"):
            parts.append(f"운영시간: {r['hours']}")
        if r.get("phone"):
            parts.append(f"전화번호: {r['phone']}")
        lines.append(" / ".join(parts))
    return "\n".join(lines)


class AnswerGenerator:
    """검색된 장소 정보(results)를 근거로 LLM이 자연어 답변을 생성한다."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def generate(self, query: str, results: List[Dict]) -> str:
        if not results:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다."

        context = _build_context(results)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"[검색된 장소 정보]\n{context}\n\n"
            f"[사용자 질문]\n{query}\n\n"
            f"[답변]"
        )
        return self.llm_client.generate(prompt)
