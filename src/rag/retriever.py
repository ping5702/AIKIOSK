from typing import Dict, List

from src.embedding.bi_encoder import BiEncoder
from src.vectorstore.faiss_store import FaissPlaceStore


class PlaceRetriever:
    """텍스트 질의를 받아 FAISS DB에서 장소 정보를 조회하고 안내 형식 텍스트로 변환한다."""

    def __init__(self, encoder: BiEncoder, store: FaissPlaceStore, top_k: int = 3, min_similarity: float = 0.0):
        self.encoder = encoder
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def search(self, query: str) -> List[Dict]:
        query_embedding = self.encoder.encode([query])
        results = self.store.search(query_embedding, self.top_k)
        results = [r for r in results if r[1] >= self.min_similarity]
        return [self._to_result(record, score) for record, score in results]

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
