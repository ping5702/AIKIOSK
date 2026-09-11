from typing import Dict, List, Tuple

from src.embedding.bi_encoder import BiEncoder
from src.rag.bm25_index import Bm25Index
from src.rag.place_name_corrector import PlaceNameCorrector
from src.vectorstore.faiss_store import FaissPlaceStore


class PlaceRetriever:
    """텍스트 질의를 받아 장소 정보를 조회하고 안내 형식 텍스트로 변환한다.

    의미 기반 벡터 검색(BiEncoder+FAISS)과 키워드 기반 BM25를 결합한 하이브리드
    검색을 쓴다 — 벡터 검색은 "쉬기 좋은 곳" 같은 문맥 질문에 강하고, BM25는
    "3층", "301호" 같은 정확한 키워드 매칭에 강해서 서로 보완한다.
    """

    def __init__(
        self,
        encoder: BiEncoder,
        store: FaissPlaceStore,
        top_k: int = 3,
        min_similarity: float = 0.0,
        bm25_weight: float = 0.3,
    ):
        self.encoder = encoder
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.bm25_weight = bm25_weight
        self.bm25_index = Bm25Index(store.metadata)
        self.name_corrector = PlaceNameCorrector(r.get("name", "") for r in store.metadata)

    def search(self, query: str) -> List[Dict]:
        # STT가 발음이 비슷한 다른 글자로 잘못 인식한 장소명(예: "해장실" -> "회장실")을
        # 알려진 장소명 목록 기준으로 미리 보정해, 임베딩/키워드 검색 둘 다 정확한 이름으로
        # 매칭되게 한다.
        query = self.name_corrector.correct(query)

        dense_scores = self._dense_scores(query)
        bm25_scores = self.bm25_index.scores(query)
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

        combined: List[Tuple[Dict, float, float]] = []
        for record, dense_score, bm25_score in zip(self.store.metadata, dense_scores, bm25_scores):
            normalized_bm25 = bm25_score / max_bm25
            combined_score = (1 - self.bm25_weight) * dense_score + self.bm25_weight * normalized_bm25
            combined.append((record, dense_score, combined_score))

        # "관련 없음" 판단 기준을 combined_score로 하되, min_similarity를 (1 - bm25_weight)만큼
        # 낮춰서 비교한다. combined_score = (1-w)*dense + w*bm25이므로, dense_score만으로
        # 통과하던 기존 케이스는 이 조정된 기준에서도 항상 그대로 통과한다(dense >= min_similarity
        # 이면 (1-w)*dense >= (1-w)*min_similarity가 항상 성립). 그 위에 BM25 키워드/음절
        # 매칭이 강한 경우(예: STT 오인식으로 dense_score는 낮지만 자모/음절이 비슷한 경우)를
        # 추가로 구제할 수 있게 된다.
        adjusted_threshold = self.min_similarity * (1 - self.bm25_weight)
        relevant = [(record, combined_score) for record, dense_score, combined_score in combined if combined_score >= adjusted_threshold]
        relevant.sort(key=lambda item: item[1], reverse=True)
        top = relevant[: self.top_k]

        return [self._to_result(record, score) for record, score in top]

    def _dense_scores(self, query: str) -> List[float]:
        query_embedding = self.encoder.encode([query])
        results = self.store.search(query_embedding, len(self.store.metadata))
        score_by_id = {id(record): score for record, score in results}
        return [score_by_id.get(id(record), 0.0) for record in self.store.metadata]

    @staticmethod
    def _to_result(record: Dict, score: float) -> Dict:
        return {
            "name": record.get("name", ""),
            "category": record.get("category", ""),
            "description": record.get("description", ""),
            "location": record.get("location", ""),
            "hours": record.get("hours", ""),
            "phone": record.get("phone", ""),
            "score": score,
        }

    def format_answer(self, results: List[Dict]) -> str:
        if not results:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다."

        best = results[0]
        lines = [f"[{best['name']}] 안내입니다."]
        if best["description"]:
            lines.append(best["description"])
        if best["location"]:
            lines.append(f"위치: {best['location']}")
        if best["hours"]:
            lines.append(f"운영시간: {best['hours']}")
        return "\n".join(lines)
