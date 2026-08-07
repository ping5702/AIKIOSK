import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

# 한글은 조사(예: "3층에"의 "에")가 붙어서 공백 기준 토큰화만으로는
# "3층"과 "3층에"가 서로 다른 토큰으로 취급된다. 2글자 이하 단어는 그대로 두고,
# 그보다 긴 단어는 글자 단위 bigram으로 쪼개서 부분 문자열이 겹치면 매칭되게 한다.
_TOKEN_PATTERN = re.compile(r"[0-9가-힣A-Za-z]+")


def tokenize(text: str) -> List[str]:
    tokens = []
    for word in _TOKEN_PATTERN.findall(text):
        if len(word) <= 2:
            tokens.append(word)
        else:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


class Bm25Index:
    """장소명, 호수 등 정확한 키워드 매칭을 위한 BM25(sparse) 스코어러.
    의미 기반 벡터 검색(BiEncoder)을 보완하는 용도로 쓴다."""

    def __init__(self, records: List[Dict]):
        self.records = records
        corpus = [tokenize(r["embedding_text"]) for r in records]
        self.bm25 = BM25Okapi(corpus)

    def scores(self, query: str) -> List[float]:
        """records와 같은 순서로, 문서별 BM25 점수를 반환한다."""
        return list(self.bm25.get_scores(tokenize(query)))
