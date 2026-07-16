import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class FaissPlaceStore:
    """정규화된 임베딩에 대해 코사인 유사도로 검색하는 FAISS 기반 벡터 저장소."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise ValueError("임베딩 개수와 메타데이터 개수가 일치하지 않습니다.")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissPlaceStore":
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.metadata = metadata
        return store
