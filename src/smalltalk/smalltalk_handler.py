from typing import Optional

# 장소 검색(RAG)과 분리된 인사/잡담 규칙. 키워드가 매칭되면 고정 응답을 반환한다.
_SMALLTALK_RULES = [
    (("안녕", "hi", "hello"), "안녕하세요! 궁금하신 시설이나 장소를 말씀해 주세요."),
    (("고마워", "감사"), "천만에요! 더 필요한 안내가 있으신가요?"),
    (("잘가", "안녕히", "bye"), "안녕히 가세요! 좋은 하루 되세요."),
]


def match_smalltalk(query: str) -> Optional[str]:
    """질문이 인사/잡담이면 고정 응답을, 아니면 None을 반환한다."""
    normalized = query.strip()
    for keywords, response in _SMALLTALK_RULES:
        if any(keyword in normalized for keyword in keywords):
            return response
    return None
